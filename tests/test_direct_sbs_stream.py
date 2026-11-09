import io
import queue
import subprocess
import sys
import threading
from pathlib import Path

from path_config import APP_ROOT
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import streaming.direct_sbs as direct_sbs
import streaming.nvidia_encoder as nvidia_encoder
from streaming.direct_sbs import (
    DirectSbsOutputConsumer,
    FfmpegDirectSbsOutput,
    IntelQsvDirectSbsOutput,
    MjpegDirectSbsOutput,
    PyNvDirectSbsOutput,
    RuntimeSbsRgbConverter,
    VulkanDirectSbsOutput,
    runtime_sbs_to_rgb,
)
from streaming.nvidia_encoder import H264MpegTsTimestampMuxer, PyNvSrtVideoOutput


def _stub_streaming_executables(monkeypatch, tmp_path):
    for name, variable in (
        ("ffmpeg", "D2S_FFMPEG_PATH"),
        ("mediamtx", "D2S_MEDIAMTX_PATH"),
    ):
        executable = tmp_path / name
        executable.touch()
        monkeypatch.setenv(variable, str(executable))


def test_runtime_sbs_to_rgb_converts_chw_float_to_hwc_uint8():
    frame = np.array(
        [
            [[0.0, 1.0], [0.5, 0.25]],
            [[1.0, 0.0], [0.5, 0.75]],
            [[0.0, 0.5], [1.0, 0.25]],
        ],
        dtype=np.float32,
    )

    rgb = runtime_sbs_to_rgb(SimpleNamespace(sbs=frame))

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb.flags.c_contiguous
    assert rgb[0, 1].tolist() == [255, 0, 128]


def test_native_stream_frame_is_packed_on_output_thread():
    shutdown = threading.Event()
    submitted = []

    class NativeFrame:
        def __init__(self):
            self.released = False

        def pack(self, destination, output_size, output_format, *, rgb=False):
            assert output_size == (4, 2)
            assert output_format == "half_tab"
            assert rgb is True
            for offset in range(0, len(destination), 3):
                destination[offset : offset + 3] = bytes((10, 20, 30))

        def release(self):
            self.released = True

    class Output:
        display_mode = "Half-TAB"
        fit_mode = "stretch"
        input_size = (4, 2)
        synchronous_submit = True

        def submit_frame(self, frame):
            submitted.append(frame)
            shutdown.set()

    native = NativeFrame()
    result = SimpleNamespace(
        native_stream_frame=native,
        native_stream_fallback=None,
        output_display_size=(4, 2),
        output_format="half_tab",
        sbs=None,
        left_eye=None,
        right_eye=None,
    )
    consumer = DirectSbsOutputConsumer(
        runtime_q=queue.Queue(),
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda *args, **kwargs: None,
    )
    consumer.runtime_q.put((result, 0.0))
    thread = threading.Thread(target=consumer.run)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert native.released is True
    assert len(submitted) == 1
    assert submitted[0].shape == (2, 4, 3)
    assert submitted[0].dtype == np.uint8
    assert submitted[0][0, 0].tolist() == [10, 20, 30]


def test_native_stream_pack_failure_uses_runtime_fallback():
    shutdown = threading.Event()
    stats = []
    submitted = []

    class NativeFrame:
        def __init__(self):
            self.released = False

        def pack(self, *args, **kwargs):
            raise RuntimeError("injected native pack failure")

        def release(self):
            self.released = True

    class Output:
        display_mode = "Half-TAB"
        fit_mode = "stretch"
        input_size = (4, 2)
        synchronous_submit = True

        def submit_frame(self, frame):
            submitted.append(frame)
            shutdown.set()

    native = NativeFrame()
    fallback = SimpleNamespace(
        sbs=np.full((2, 4, 3), 77, dtype=np.uint8),
        left_eye=None,
        right_eye=None,
        native_stream_frame=None,
    )
    result = SimpleNamespace(
        native_stream_frame=native,
        native_stream_fallback=lambda: fallback,
        output_display_size=(4, 2),
        output_format="half_tab",
        sbs=None,
        left_eye=None,
        right_eye=None,
    )
    consumer = DirectSbsOutputConsumer(
        runtime_q=queue.Queue(),
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda name, **kwargs: stats.append(name),
    )
    consumer.runtime_q.put((result, 0.0))
    thread = threading.Thread(target=consumer.run)
    thread.start()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert native.released is True
    assert submitted[0][0, 0].tolist() == [77, 77, 77]
    assert "native_stream_pack_error" in stats
    assert "native_stream_fallback" in stats


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_runtime_sbs_to_rgb_converts_mps_tensor_to_host():
    # MJPEG on macOS feeds MPS SBS tensors; numpy() without .cpu() raises
    # "can't convert mps:0 device type tensor to numpy".
    import torch

    frame = torch.rand(1, 3, 64, 96, device="mps")

    rgb = runtime_sbs_to_rgb(SimpleNamespace(sbs=frame))

    assert rgb.shape == (64, 96, 3)
    assert rgb.dtype == np.uint8

    packed = (torch.rand(1, 3, 64, 96, device="mps") * 255).to(torch.uint8)
    rgb8 = runtime_sbs_to_rgb(SimpleNamespace(sbs=packed))
    assert rgb8.shape == (64, 96, 3)
    assert rgb8.dtype == np.uint8


def test_darwin_audio_input_args_strip_backend_prefix(monkeypatch) -> None:
    # macOS RTMP maps "soundcard:BlackHole 2ch" to an avfoundation input; the
    # prefix must be stripped or ffmpeg fails ("Error opening input file
    # :soundcard:BlackHole 2ch"). A configured NAME is resolved to its
    # AVFoundation index (v2.5.0 get_device_index parity).
    target = object.__new__(FfmpegDirectSbsOutput)
    target.os_name = "Darwin"
    target.audio_delay = 0.0
    target.ffmpeg_path = Path("/usr/bin/ffmpeg")
    monkeypatch.setattr(direct_sbs, "_auto_select_darwin_audio", lambda ff: "1")
    monkeypatch.setattr(
        direct_sbs,
        "_list_darwin_audio_devices",
        lambda ff: [(0, "BlackHole 2ch"), (2, "Stereo Mix"), (8, "Virtual Desktop Speakers")],
    )

    target.stereo_mix_device = "soundcard:BlackHole 2ch"
    args = target._audio_input_args()
    assert args[-2:] == ["-i", ":0"]

    target.stereo_mix_device = "wasapi:Stereo Mix"
    assert target._audio_input_args()[-2:] == ["-i", ":2"]

    # A stale/unplugged configured name falls back to auto-selection instead
    # of handing FFmpeg a device that no longer exists.
    target.stereo_mix_device = "soundcard:Gone Device"
    assert target._audio_input_args()[-2:] == ["-i", ":1"]

    # Empty / bare-prefix / lone-colon / "no device" values auto-select the
    # available macOS loopback device instead of silently dropping audio.
    for bad in ("", "soundcard:", "wasapi:", ":", "No Stereo Mix device found"):
        target.stereo_mix_device = bad
        args = target._audio_input_args()
        assert args[-2:] == ["-i", ":1"], f"device {bad!r} did not auto-select"


def test_darwin_unconfigured_audio_auto_selects_loopback(monkeypatch) -> None:
    target = object.__new__(FfmpegDirectSbsOutput)
    target.os_name = "Darwin"
    target.audio_delay = 0.0
    target.ffmpeg_path = Path("/usr/bin/ffmpeg")
    target.stereo_mix_device = ""

    monkeypatch.setattr(direct_sbs, "_auto_select_darwin_audio", lambda _ffmpeg: "1")

    assert target._audio_input_args()[-2:] == ["-i", ":1"]


def test_darwin_auto_select_prefers_virtual_desktop_speakers(monkeypatch) -> None:
    monkeypatch.setattr(
        direct_sbs,
        "_list_darwin_audio_devices",
        lambda _ffmpeg: [
            (1, "BlackHole 2ch"),
            (8, "Virtual Desktop Speakers"),
        ],
    )

    assert direct_sbs._auto_select_darwin_audio(Path("ffmpeg")) == "8"


def test_darwin_frame_submit_replaces_stale_frame_without_blocking() -> None:
    target = object.__new__(FfmpegDirectSbsOutput)
    target._darwin_async_submit = True
    target._async_frame_queue = queue.Queue(maxsize=1)
    target._async_frame_lock = threading.Lock()
    target._async_frame_drops = 0

    first = np.zeros((2, 4, 3), dtype=np.uint8)
    second = np.full_like(first, 17)
    third = np.full_like(first, 33)
    target._async_frame_queue.put_nowait(first)

    target._enqueue_async_frame(second)
    target._enqueue_async_frame(third)

    queued = target._async_frame_queue.get_nowait()
    assert queued[0, 0].tolist() == [33, 33, 33]
    assert target._async_frame_drops == 2
    # The queued frame owns its storage, so a converter buffer can be reused.
    third.fill(99)
    assert queued[0, 0].tolist() == [33, 33, 33]


def test_darwin_submit_frame_does_not_write_on_runtime_thread() -> None:
    target = object.__new__(FfmpegDirectSbsOutput)
    target._darwin_async_submit = True
    target._async_frame_queue = queue.Queue(maxsize=1)
    target._async_frame_lock = threading.Lock()
    target._async_frame_drops = 0
    target._async_frame_error = None
    target._apply_pending_audio_delay = lambda: None
    target._calibration_controller = None
    target.ffmpeg_process = SimpleNamespace()
    target._frame_size = (4, 2)

    writes = []
    target._write_frame = lambda frame: writes.append(frame)
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    target.submit_frame(frame)

    assert writes == []
    assert target._async_frame_queue.qsize() == 1


def test_darwin_safe_mediamtx_config_zeroes_udp_read_buffer(tmp_path: Path) -> None:    # MediaMTX aborts on macOS when udpReadBufferSize > 0 ("read buffer size
    # is unimplemented"); the darwin launcher must rewrite it to 0 while
    # keeping every other setting intact.
    cfg = tmp_path / "mediamtx.yml"
    cfg.write_text(
        "udpReadBufferSize: 4194304\nwriteQueueSize: 2048\npaths:\n  all_others:\n",
        encoding="utf-8",
    )

    patched = direct_sbs._darwin_safe_mediamtx_config(cfg)

    assert patched != cfg
    text = patched.read_text(encoding="utf-8")
    assert "udpReadBufferSize: 0" in text
    assert "writeQueueSize: 2048" in text
    # Already-default configs are returned unchanged (no copy created).
    cfg2 = tmp_path / "mediamtx_default.yml"
    cfg2.write_text("udpReadBufferSize: 0\n", encoding="utf-8")
    assert direct_sbs._darwin_safe_mediamtx_config(cfg2) is cfg2


def test_pynv_backend_uses_shared_ffmpeg_probe_for_calibration(monkeypatch):
    calls = []

    output = object.__new__(PyNvDirectSbsOutput)
    output._calibration_controller = object()

    def fake_base_start(self, width, height):
        calls.append((width, height))

    monkeypatch.setattr(FfmpegDirectSbsOutput, "_start_ffmpeg", fake_base_start)
    direct_sbs._PyNvDirectSbsOutputMixin._start_ffmpeg(output, 1920, 1080)

    assert calls == [(1920, 1080)]


def test_runtime_sbs_to_rgb_drops_alpha_from_hwc_uint8():
    frame = np.zeros((2, 3, 4), dtype=np.uint8)
    frame[..., 0] = 7
    frame[..., 3] = 255

    rgb = runtime_sbs_to_rgb(frame)

    assert rgb.shape == (2, 3, 3)
    assert np.all(rgb[..., 0] == 7)


def test_direct_consumer_submits_latest_sbs_frame():
    runtime_q = queue.Queue()
    shutdown = threading.Event()
    submitted = []
    stats = []

    class Output:
        def submit_frame(self, frame):
            submitted.append(frame.copy())
            shutdown.set()

    runtime_q.put((SimpleNamespace(sbs=np.zeros((3, 2, 2))), 1.0))
    runtime_q.put((SimpleNamespace(sbs=np.ones((3, 2, 2))), 2.0))
    consumer = DirectSbsOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda name, *args, **kwargs: stats.append(name),
    )

    consumer.run()

    assert len(submitted) == 1
    # The NEWEST (ones) frame is the one submitted. The 2x2 source is not
    # 16:9, so the contain-fit transport canvas letterboxes it into black
    # bars; only the content pixels are 255, and the stale zeros frame must
    # not have been submitted.
    assert np.any(submitted[0] == 255)
    assert not np.all(submitted[0] == 0)
    assert "runtime_output_overwrite" in stats
    assert "network_stream_frames" in stats


def test_direct_consumer_routes_deferred_vulkan_request_to_native_sink():
    runtime_q = queue.Queue()
    shutdown = threading.Event()
    submitted = []

    class Output:
        def submit_vulkan_stereo_frame(self, runtime_result):
            submitted.append(runtime_result.vulkan_compute_request)
            shutdown.set()

    request = object()
    runtime_q.put((SimpleNamespace(vulkan_compute_request=request), 1.0))
    consumer = DirectSbsOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda *args, **kwargs: None,
    )

    consumer.run()

    assert submitted == [request]


def test_direct_consumer_drops_frame_before_conversion_when_not_due():
    runtime_q = queue.Queue()
    shutdown = threading.Event()
    submitted = []

    class Output:
        def should_submit_frame(self, now):
            shutdown.set()
            return False

        def submit_frame(self, frame):
            submitted.append(frame)

    runtime_q.put((SimpleNamespace(sbs=object()), 1.0))
    consumer = DirectSbsOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda *args, **kwargs: None,
    )

    consumer.run()

    assert submitted == []


def test_direct_consumer_logs_fps_when_enabled(capsys):
    runtime_q = queue.Queue()
    shutdown = threading.Event()
    clock_values = iter((0.0, 0.1, 0.2, 0.3, 0.4, 5.0))

    class Output:
        def submit_frame(self, frame):
            shutdown.set()

    runtime_q.put((SimpleNamespace(sbs=np.ones((3, 2, 2))), 1.0))
    consumer = DirectSbsOutputConsumer(
        runtime_q=runtime_q,
        shutdown_event=shutdown,
        output=Output(),
        source_stat_inc=lambda *args, **kwargs: None,
        show_fps_provider=lambda: True,
        clock=lambda: next(clock_values),
    )

    consumer.run()

    assert (
        "[DirectSbsStream] SBS FPS: 0.2 network_bitrate=0.0 Mbps submitted=0.2 "
        "convert_ms=100.0 submit_ms=100.0"
        in capsys.readouterr().out
    )


def test_direct_consumer_logs_current_network_bitrate(capsys):
    consumer = DirectSbsOutputConsumer(
        runtime_q=queue.Queue(),
        shutdown_event=threading.Event(),
        output=SimpleNamespace(current_network_bitrate_mbps=42.5),
        source_stat_inc=lambda *args, **kwargs: None,
        show_fps_provider=lambda: True,
        fps_report_interval=1.0,
        clock=lambda: 0.0,
    )
    consumer._fps_sbs_frames = 30
    consumer._fps_submitted_frames = 30
    consumer._clock = lambda: 1.0

    consumer._report_fps_if_due()

    assert "SBS FPS: 30.0 network_bitrate=42.5 Mbps" in capsys.readouterr().out


def test_direct_consumer_hides_fps_when_disabled(capsys):
    samples = []
    calibration_samples = []
    consumer = DirectSbsOutputConsumer(
        runtime_q=queue.Queue(),
        shutdown_event=threading.Event(),
        output=SimpleNamespace(
            observe_calibration_window=lambda **values: calibration_samples.append(values)
        ),
        source_stat_inc=lambda *args, **kwargs: None,
        show_fps_provider=lambda: False,
        on_sbs_fps=lambda fps, **kwargs: samples.append((fps, kwargs)),
        fps_report_interval=1.0,
        clock=lambda: 0.0,
    )
    consumer._fps_sbs_frames = 30
    consumer._fps_submitted_frames = 29
    consumer._clock = lambda: 1.0

    consumer._report_fps_if_due()

    assert "SBS FPS" not in capsys.readouterr().out
    assert samples == [(30.0, {"frame_count": 30})]
    assert calibration_samples == [{
        "sbs_fps": 30.0,
        "submitted_fps": 29.0,
        "convert_ms": 0.0,
        "submit_ms": 0.0,
    }]


def test_cuda_converter_reuses_pinned_host_buffer():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    converter = RuntimeSbsRgbConverter(fit_mode="stretch")
    first_tensor = torch.zeros((3, 2, 4), device="cuda", dtype=torch.uint8)
    second_tensor = torch.full(
        (3, 2, 4), 255, device="cuda", dtype=torch.uint8
    )

    first = converter.convert(first_tensor).copy()
    host_pointer = converter._host_rgb.data_ptr()
    second = converter.convert(second_tensor)

    assert first.shape == (2, 4, 3)
    assert np.all(first == 0)
    assert np.all(second == 255)
    assert converter._host_rgb.data_ptr() == host_pointer
    assert converter._host_rgb.is_pinned()


def test_auto_stream_attempts_vendor_gpu_before_loading_vulkan():
    output = object.__new__(VulkanDirectSbsOutput)
    output._vendor_gpu_first = True
    output._opengl_fallback_attempted = False
    output._native_active = False
    output._host_fallback = None
    calls = []

    def activate(frame, reason, *, vendor_first=False):
        calls.append((frame, str(reason), vendor_first))
        return True

    output._fallback_to_opengl = activate
    frame = object()

    output.submit_cuda_frame(frame)

    assert calls == [(frame, "Auto vendor-native GPU stage", True)]


def test_vulkan_output_disables_native_bridge_under_validation(monkeypatch, capsys):
    monkeypatch.setenv("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation")
    output = VulkanDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="WEBRTC",
        port=1122,
        stream_key="live",
        fps=30,
        crf=23,
        os_name="Windows",
        stereo_mix_device="",
    )

    assert output._native_vulkan_bridge is None
    assert "disabled under VK_LAYER_KHRONOS_validation" in capsys.readouterr().out


def test_vulkan_native_mux_uses_wallclock_sync_and_async1(monkeypatch):
    captured = {}

    class RunningProcess:
        stdin = object()
        stderr = io.StringIO()
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return RunningProcess()

    output = object.__new__(VulkanDirectSbsOutput)
    output.os_name = "Windows"
    output.protocol = "WEBRTC"
    output.fps = 30
    output.stream_key = "live"
    output.ffmpeg_path = Path("ffmpeg")
    output.stereo_mix_device = "soundcard:Stereo Mix"
    output.audio_delay = 0.0
    output._soundcard_audio = None
    monkeypatch.setattr(VulkanDirectSbsOutput, "_native_output_url", lambda self: "rtsp://127.0.0.1:8554/live")
    monkeypatch.setattr(VulkanDirectSbsOutput, "_audio_input_args", lambda self: ["-itsoffset", "0.0", "-f", "dshow", "-i", "audio=Stereo Mix"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    output._start_native_mux()

    command = captured["command"]
    # The native mux carries the same professional A/V sync as the FFmpeg
    # path: wall-clock timestamps on both inputs, threaded demux, async=1.
    assert command.count("-use_wallclock_as_timestamps") == 2
    assert command.count("-thread_queue_size") == 2
    assert command[command.index("-af") + 1] == "aresample=async=1"
    assert command[command.index("-c:a") + 1] == "libopus"


def test_packet_loss_detector_matches_transport_loss_messages():
    assert FfmpegDirectSbsOutput._looks_like_packet_loss(
        "[WebRTC] packet missed: sequence gap"
    )
    assert FfmpegDirectSbsOutput._looks_like_packet_loss(
        "RTP buffer underflow"
    )
    assert not FfmpegDirectSbsOutput._looks_like_packet_loss(
        "[WebRTC] peer connected"
    )


def test_streaming_runtime_dir_overrides_bundled_paths(monkeypatch, tmp_path):
    runtime_root = tmp_path / "streaming"
    ffmpeg_bin = runtime_root / "ffmpeg" / "bin"
    mediamtx_dir = runtime_root / "mediamtx"
    ffmpeg_bin.mkdir(parents=True)
    mediamtx_dir.mkdir(parents=True)
    (ffmpeg_bin / "ffmpeg").write_bytes(b"ffmpeg")
    (mediamtx_dir / "mediamtx").write_bytes(b"mediamtx")
    (runtime_root / "mediamtx.yml").write_text("paths: {}", encoding="utf-8")
    monkeypatch.setenv("D2S_STREAMING_RUNTIME_DIR", str(runtime_root))

    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT), protocol="RTMP", port=1935, stream_key="live",
        fps=30, crf=20, os_name="Linux"
    )

    assert output.ffmpeg_path == (ffmpeg_bin / "ffmpeg").resolve()
    assert output.mediamtx_path == (mediamtx_dir / "mediamtx").resolve()
    assert output.mediamtx_config == (runtime_root / "mediamtx.yml").resolve()


def test_ffmpeg_output_finds_bundled_encoder_and_config():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    assert output.ffmpeg_path.name == "ffmpeg.exe"
    assert output.mediamtx_path.name == "mediamtx.exe"
    if sys.platform == "darwin":
        # The darwin launcher substitutes a runtime-generated config with the
        # OS-default UDP read buffer (udpReadBufferSize is unimplemented on
        # macOS) even when a Windows os_name is simulated for the test.
        assert output.mediamtx_config.name == "mediamtx.macos.yml"
    else:
        assert output.mediamtx_config.name == "mediamtx.yml"
    assert output.publish_rtsp_port == 8554
    server_env = output._server_environment()
    assert server_env["MTX_RTMPADDRESS"] == ":1935"
    assert server_env["MTX_METRICS"] == "yes"
    assert server_env["MTX_METRICSADDRESS"] == "127.0.0.1:9998"


def test_mediamtx_startup_error_includes_server_output(monkeypatch):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    class FailedProcess:
        returncode = 1
        stdout = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            return (
                "2026/08/19 18:57:14 ERR listen udp :8000: bind: address in use\n",
                None,
            )

    monkeypatch.setattr(
        direct_sbs.subprocess, "Popen", lambda *args, **kwargs: FailedProcess()
    )
    monkeypatch.setattr(direct_sbs.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match=r"listen udp :8000: bind"):
        output.start()


def test_non_webrtc_start_recommends_webrtc(monkeypatch, capsys):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=23,
        os_name="Windows",
    )

    class RunningProcess:
        stdout = None

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        direct_sbs.subprocess, "Popen", lambda *args, **kwargs: RunningProcess()
    )
    monkeypatch.setattr(direct_sbs.time, "sleep", lambda _seconds: None)

    output.start()

    message = capsys.readouterr().out
    assert "WARNING: RTMP selected" in message
    assert "WebRTC is recommended" in message


def test_stream_rate_uses_stable_windows_and_fixed_pacing():
    selected_fps = []
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=60,
        crf=20,
        os_name="Windows",
        on_stream_fps_selected=selected_fps.append,
    )

    first_submit_at = None
    for index in range(241):
        timestamp = index / 40.0
        if output.should_submit_frame(timestamp):
            first_submit_at = timestamp
            break

    assert first_submit_at is not None
    assert first_submit_at >= 5.0
    assert output.fps == 30
    assert selected_fps == [30]
    assert not output.should_submit_frame(first_submit_at + 0.01)
    assert output.should_submit_frame(first_submit_at + 0.04)


def test_stream_rate_selector_keeps_headroom_across_gpu_speeds():
    select = FfmpegDirectSbsOutput._select_sustainable_stream_fps

    assert select(40.0, 60) == 30
    assert select(43.0, 60) == 42
    assert select(44.3, 60) == 42
    assert select(30.1, 30) == 30
    assert select(29.9, 30) == 25
    assert select(28.0, 60) == 25
    assert select(20.0, 60) == 15


def test_unstable_stream_rate_uses_low_recent_percentile():
    sample = FfmpegDirectSbsOutput._fallback_rate_sample

    assert sample([42.0, 35.0, 41.0, 36.0, 40.0, 37.0]) == 36.0


def test_mediamtx_metrics_extract_actual_path_inbound_bytes():
    metrics = """
paths_inbound_bytes{name="live",state="ready"} 5898240
paths_outbound_bytes{name="live",state="ready"} 4718592
"""

    assert FfmpegDirectSbsOutput._parse_mediamtx_path_inbound_bytes(
        metrics, "live"
    ) == 5898240
    assert FfmpegDirectSbsOutput._parse_mediamtx_path_inbound_bytes(
        metrics, "other"
    ) is None


def test_windows_rtmp_command_keeps_legacy_srt_transport_parameters():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="legacy-compatible",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    command = output._ffmpeg_command(1920, 1080)

    assert command[command.index("-f") + 1] == "rawvideo"
    assert "pipe:0" in command
    assert "gfxcapture" not in " ".join(command)
    assert "libx264" in command
    assert "ultrafast" in command
    assert "zerolatency" in command
    x264_params = command[command.index("-x264-params") + 1]
    assert "keyint=30:min-keyint=30" in x264_params
    assert "scenecut=0" in x264_params
    assert "open-gop=0" in x264_params
    assert "repeat-headers=1" in x264_params
    assert "mpegts" in command
    assert (
        "srt://127.0.0.1:8890?"
        "streamid=publish:legacy-compatible&pkt_size=1316"
    ) == command[-1]


def test_encoder_candidates_cover_platform_hardware_fallbacks(monkeypatch):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    monkeypatch.setattr(direct_sbs.platform, "system", lambda: "Windows")
    assert [name for name, _label in output._encoder_candidates()] == [
        "h264_nvenc",
        "h264_qsv",
        "h264_amf",
        "libx264",
    ]

    linux_output = object.__new__(FfmpegDirectSbsOutput)
    linux_output.os_name = "Linux"
    linux_output.use_hevc = False
    linux_output.prefer_nvenc = False
    assert [name for name, _label in linux_output._encoder_candidates()] == [
        "h264_qsv", "h264_vaapi", "libx264"
    ]

    mac_output = object.__new__(FfmpegDirectSbsOutput)
    mac_output.os_name = "Darwin"
    mac_output.use_hevc = False
    mac_output.prefer_nvenc = False
    assert [name for name, _label in mac_output._encoder_candidates()] == [
        "h264_videotoolbox", "libx264"
    ]


def test_hardware_encoder_failure_falls_back_to_software(monkeypatch):
    output = object.__new__(FfmpegDirectSbsOutput)
    output.os_name = "Windows"
    output.use_hevc = False
    output.prefer_nvenc = True
    monkeypatch.setattr(output, "_probe_encoder", lambda *_args, **_kwargs: False)

    assert output._select_video_encoder(3840, 2160) == "libx264"
    assert output._encoder_selection_reason == "software fallback"


def test_windows_soundcard_audio_uses_wasapi_loopback_sender(monkeypatch) -> None:
    # NVIDIA/ROCm run on Windows with the "soundcard:" speaker label; it is
    # captured through the python soundcard WASAPI loopback (UDP s16le), not
    # dshow. The macOS avfoundation parity work must not alter this path.
    output = object.__new__(FfmpegDirectSbsOutput)
    output.os_name = "Windows"
    output.stereo_mix_device = "soundcard:Stereo Mix (Realtek(R) Audio)"
    output.audio_delay = -0.15
    output._soundcard_audio = None
    sender = SimpleNamespace(
        ffmpeg_url="udp://127.0.0.1:54321", start=lambda: None
    )
    monkeypatch.setattr(direct_sbs, "SoundcardLoopbackSender", lambda name: sender)

    args = output._audio_input_args()

    assert args == [
        "-itsoffset", "-0.15",
        "-f", "s16le", "-ar", "48000", "-ac", "2",
        "-i", "udp://127.0.0.1:54321",
    ]
    assert output._soundcard_audio is sender


def test_windows_rtmp_audio_keeps_aac_for_nvidia_rocm():
    # Windows/Linux RTMP (the SRT ingest used by NVIDIA/ROCm) keeps AAC; only
    # the macOS RTSP/Opus path was aligned with v2.5.0. A bare dshow-style
    # device label exercises the non-loopback Windows input.
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        stereo_mix_device="virtual-audio-capturer",
    )

    command = output._ffmpeg_command(3840, 1080)

    assert command[command.index("-c:a") + 1] == "aac"
    assert "libopus" not in command
    # The audio timeline is normalized on every platform (wall-clock inputs +
    # v2.5.0 async=1 drift absorption); the codec stays AAC for NVIDIA/ROCm.
    assert command[command.index("-af") + 1] == "aresample=async=1"


def test_macos_audio_uses_avfoundation_device_index():
    output = object.__new__(FfmpegDirectSbsOutput)
    output.os_name = "Darwin"
    output.stereo_mix_device = "2"
    output.audio_delay = -0.1

    # v2.5.0 parity: the AVFoundation audio input keeps the 256 MB ring
    # buffer (rtbufsize) used by the reference macOS build.
    assert output._audio_input_args() == [
        "-itsoffset", "-0.1", "-f", "avfoundation", "-rtbufsize", "256M", "-i", ":2"
    ]


def test_macos_rtmp_stream_audio_uses_opus_like_v250(monkeypatch, tmp_path) -> None:
    # v2.5.0 macOS parity: every non-WebRTC client protocol (RTMP/RTSP/HLS)
    # publishes Opus 96k/48k to MediaMTX's RTSP ingest; AAC is only used on
    # the Windows/Linux SRT/RTMP paths (NVIDIA/ROCm soundcard chain).
    _stub_streaming_executables(monkeypatch, tmp_path)
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Darwin",
        stereo_mix_device="soundcard:Virtual Desktop Speakers",
        audio_delay=-0.1,
    )
    monkeypatch.setattr(
        direct_sbs,
        "_list_darwin_audio_devices",
        lambda ff: [(8, "Virtual Desktop Speakers")],
    )

    command = output._ffmpeg_command(3840, 1080)

    assert command[command.index("-c:a") + 1] == "libopus"
    # macOS uses a relative audio clock so the bundled FFmpeg emits packets;
    # the configured delay is folded into the filter graph.
    assert command[command.index("-af") + 1] == (
        "asetpts=PTS-STARTPTS-100000/(1000000*TB),aresample=async=1"
    )
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-b:a") + 1] == "96k"
    assert command[command.index("-rtbufsize") + 1] == "256M"
    assert "aac" not in command
    # The device NAME is resolved to its AVFoundation index; the audio
    # input's "-i" follows "-rtbufsize" (the video input is rawvideo pipe:0).
    audio_i = command.index("-i", command.index("-rtbufsize"))
    assert command[audio_i + 1] == ":8"
    # Professional A/V sync: only the VIDEO input uses the demuxer wall-clock
    # option (audio is re-anchored in the filter graph — see _audio_filter_graph)
    # and each input demuxes on its own thread (audio can't be starved by a
    # stalled video pipe producer).
    assert command.count("-use_wallclock_as_timestamps") == 1
    assert command.count("-thread_queue_size") == 2
    assert command[command.index("-use_wallclock_as_timestamps") + 1] == "1"
    # Immediate mux flushing keeps audio RTP delivery from batching into
    # bursts when the app-paced video stalls.
    assert "-flush_packets" in command


def test_macos_webrtc_stream_audio_uses_opus_async1(monkeypatch, tmp_path) -> None:
    _stub_streaming_executables(monkeypatch, tmp_path)
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="WebRTC",
        port=1122,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Darwin",
        stereo_mix_device="soundcard:BlackHole 2ch",
    )
    monkeypatch.setattr(
        direct_sbs,
        "_list_darwin_audio_devices",
        lambda ff: [(1, "BlackHole 2ch")],
    )

    command = output._ffmpeg_command(3840, 1080)

    assert command[command.index("-c:a") + 1] == "libopus"
    # Default audio_delay (-0.1s) folds into the relative macOS timestamp.
    assert command[command.index("-af") + 1] == (
        "asetpts=PTS-STARTPTS-100000/(1000000*TB),aresample=async=1"
    )
    assert "aac" not in command


def test_qsv_and_amf_commands_avoid_nvenc_only_options():
    output = object.__new__(FfmpegDirectSbsOutput)
    output.os_name = "Windows"
    output.protocol = "RTMP"
    output.fps = 30
    output.crf = 23
    output.use_hevc = False
    output.video_encoder = "h264_qsv"
    output.stream_key = "live"
    output._active_rate_budget = None
    output.stereo_mix_device = ""
    output.audio_delay = -0.1
    output.ffmpeg_path = Path("ffmpeg")
    qsv_command = output._ffmpeg_command(1920, 1080)
    assert "-global_quality" in qsv_command
    assert "-look_ahead" in qsv_command
    assert "-cq" not in qsv_command

    output.video_encoder = "h264_amf"
    amf_command = output._ffmpeg_command(1920, 1080)
    assert "ultralowlatency" in amf_command
    assert "vbr_peak" in amf_command
    assert "-cq" not in amf_command


def test_intel_qsv_backend_prefers_qsv_and_enables_d3d11_upload(monkeypatch):
    output = object.__new__(IntelQsvDirectSbsOutput)
    output.use_hevc = False
    output.os_name = "Windows"
    output.video_encoder = "h264_qsv"
    monkeypatch.delenv("D2S_QSV_D3D11_UPLOAD", raising=False)

    assert output._encoder_candidates() == [
        ("h264_qsv", "Intel Quick Sync/D3D11"),
        ("libx264", "software fallback"),
    ]
    assert output._qsv_d3d11_surface_upload_enabled() is True


def test_qsv_can_use_opt_in_d3d11_surface_upload(monkeypatch):
    output = object.__new__(FfmpegDirectSbsOutput)
    output.os_name = "Windows"
    output.protocol = "RTMP"
    output.fps = 30
    output.crf = 23
    output.use_hevc = False
    output.video_encoder = "h264_qsv"
    output.stream_key = "live"
    output._active_rate_budget = None
    output.stereo_mix_device = ""
    output.audio_delay = -0.1
    output.ffmpeg_path = Path("ffmpeg")
    monkeypatch.setenv("D2S_QSV_D3D11_UPLOAD", "1")
    monkeypatch.setenv("D2S_QSV_D3D11_ADAPTER", "2")

    command = output._ffmpeg_command(1920, 1080)

    assert "d3d11va=d2s_d3d11:2" in command
    assert "qsv=d2s_qsv@d2s_d3d11" in command
    assert command[command.index("-filter_hw_device") + 1] == "d2s_qsv"
    assert command[command.index("-vf") + 1] == (
        "format=nv12,hwupload=extra_hw_frames=16,"
        "hwmap=derive_device=qsv,format=qsv"
    )
    assert "-pix_fmt" not in command
    assert output._qsv_surface_mode == "d3d11_upload"


def test_vaapi_command_uploads_frames_to_hardware():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
    )
    output.video_encoder = "h264_vaapi"

    command = output._ffmpeg_command(1920, 1080)

    assert command[command.index("-c:v") + 1] == "h264_vaapi"
    assert command[command.index("-vf") + 1] == "format=nv12,hwupload"


def test_nvenc_command_uses_low_latency_hardware_encoder():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
    )
    output.video_encoder = "h264_nvenc"

    command = output._ffmpeg_command(3840, 2160)

    assert "h264_nvenc" in command
    assert "libx264" not in command
    assert command[command.index("-preset") + 1] == "p1"
    assert command[command.index("-cq") + 1] == "20"
    assert command[command.index("-zerolatency") + 1] == "1"
    assert command[command.index("-forced-idr") + 1] == "1"
    assert command[command.index("-strict_gop") + 1] == "1"
    # The FFmpeg path applies the same wall-clock A/V sync as the NVIDIA SRT
    # path: the video input is timestamped at read time so its PTS tracks the
    # real-time audio clock (no audio device here, so only the video input).
    assert command.count("-use_wallclock_as_timestamps") == 1
    # Frame pacing stays the app's job; fps_mode is not imposed.
    assert "-fps_mode" not in command


def test_calibration_uses_independent_30_fps_cbr_pressure_stream():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="WEBRTC",
        port=1122,
        stream_key="live",
        fps=30,
        crf=23,
        os_name="Windows",
        target_bitrate_mbps=30,
        peak_bitrate_mbps=34,
    )
    output._calibration_controller = SimpleNamespace()
    output.video_encoder = "h264_nvenc"

    command = output._ffmpeg_command(3840, 2160)

    assert "lavfi" in command
    assert "testsrc2=size=3840x2160:rate=30" in command
    assert "rawvideo" not in command
    assert "pipe:0" not in command
    assert command[command.index("-rc") + 1] == "cbr"
    assert "-cq" not in command
    assert command[command.index("-b:v") + 1] == "30M"
    assert command[command.index("-minrate") + 1] == "30M"
    assert command[command.index("-maxrate") + 1] == "30M"
    assert command[command.index("-bufsize") + 1] == "30M"


def test_full_sbs_uses_hevc_nvenc_without_resizing(monkeypatch):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="HLS",
        port=8888,
        stream_key="live",
        fps=60,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
        display_mode="Full-SBS",
    )
    monkeypatch.setattr(output, "_probe_nvenc", lambda width, height: True)

    assert output._select_video_encoder(7680, 2160) == "hevc_nvenc"
    output.video_encoder = "hevc_nvenc"
    command = output._ffmpeg_command(7680, 2160)

    assert "hevc_nvenc" in command
    assert "h264_nvenc" not in command
    assert "7680x2160" in command
    assert "-vf" not in command
    assert command[command.index("-b:v") + 1] == "75M"
    assert command[command.index("-maxrate") + 1] == "87M"
    assert command[command.index("-bufsize") + 1] == "87M"
    assert command[command.index("-spatial-aq") + 1] == "1"
    assert command[command.index("-temporal-aq") + 1] == "1"


def test_full_sbs_hevc_falls_back_to_libx265(monkeypatch, capsys):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="HLS",
        port=8888,
        stream_key="live",
        fps=60,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
        display_mode="Full-SBS",
    )
    monkeypatch.setattr(output, "_probe_nvenc", lambda width, height: False)
    monkeypatch.setattr(output, "_probe_encoder", lambda *_args: False)

    assert output._select_video_encoder(7680, 2160) == "libx265"
    assert "falling back to libx265" in capsys.readouterr().out
    output.video_encoder = "libx265"
    command = output._ffmpeg_command(7680, 2160)
    assert "libx265" in command
    assert "-x265-params" in command
    assert command[command.index("-maxrate") + 1] == "87M"
    assert command[command.index("-bufsize") + 1] == "87M"


def test_dynamic_stream_quality_tracks_codec_fps_resolution_and_crf():
    h264 = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="HLS",
        port=8888,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        display_mode="Half-SBS",
    )
    hevc_lower_quality = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="HLS",
        port=8888,
        stream_key="live",
        fps=60,
        crf=30,
        os_name="Windows",
        display_mode="Full-SBS",
    )

    assert h264._dynamic_stream_rate_budget(3840, 2160) == (30, 35, 35)
    assert hevc_lower_quality._dynamic_stream_rate_budget(7680, 2160) == (
        42,
        49,
        49,
    )


def test_calibrated_bitrate_overrides_dynamic_rate_budget():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="WEBRTC",
        port=1122,
        stream_key="live",
        fps=40,
        crf=23,
        os_name="Windows",
        display_mode="Half-SBS",
        target_bitrate_mbps=24,
        peak_bitrate_mbps=28,
    )

    assert output._dynamic_stream_rate_budget(3840, 2160) == (24, 28, 28)


@pytest.mark.parametrize("os_name", ["Windows", "Linux", "Darwin"])
def test_manual_bitrate_budget_is_shared_across_platforms(os_name):
    # The rate policy is shared and does not need a platform encoder binary.
    output = object.__new__(FfmpegDirectSbsOutput)
    output.os_name = os_name
    output.protocol = "WEBRTC"
    output.target_bitrate_mbps = 25
    output.peak_bitrate_mbps = 35

    assert output._dynamic_stream_rate_budget(3840, 2160) == (25, 35, 35)


def test_dynamic_rate_budget_covers_rtmp_and_webrtc_but_not_rtsp():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=60,
        crf=20,
        os_name="Windows",
        display_mode="Full-SBS",
    )

    assert output._dynamic_stream_rate_budget(7680, 2160) == (75, 87, 87)
    command = output._ffmpeg_command(7680, 2160)
    assert command[command.index("-maxrate") + 1] == "87M"
    assert command[command.index("-bufsize") + 1] == "87M"

    output.protocol = "WEBRTC"
    output.display_mode = "Half-SBS"
    output.use_hevc = False
    output.fps = 50
    output.crf = 23
    output.video_encoder = "h264_nvenc"
    command = output._ffmpeg_command(3840, 2160)
    assert command[command.index("-b:v") + 1] == "42M"
    assert command[command.index("-maxrate") + 1] == "49M"
    assert command[command.index("-bufsize") + 1] == "49M"
    assert command[command.index("-g") + 1] == "50"
    # Frame-based keyframe cadence: time expressions (gte(t,...)) break once
    # the input carries wall-clock PTS (t would force every frame).
    assert command[command.index("-force_key_frames") + 1] == "expr:eq(mod(n,50),0)"
    assert command[command.index("-pkt_size") + 1] == "1452"
    assert command[-1] == "rtsp://127.0.0.1:8554/live?pkt_size=1452"

    output.protocol = "RTSP"
    command = output._ffmpeg_command(3840, 2160)
    assert "-maxrate" not in command
    assert "-bufsize" not in command


def test_windows_stream_audio_uses_hls_compatible_aac():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="HLS",
        port=8888,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        stereo_mix_device="virtual-audio-capturer",
    )

    command = output._ffmpeg_command(3840, 1080)

    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-b:a") + 1] == "128k"
    assert "libopus" not in command


def test_windows_webrtc_stream_audio_uses_opus():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="WebRTC",
        port=8889,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        stereo_mix_device="virtual-audio-capturer",
    )

    command = output._ffmpeg_command(3840, 1080)

    assert command[command.index("-c:a") + 1] == "libopus"
    assert command[command.index("-af") + 1] == "aresample=async=1"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-b:a") + 1] == "96k"
    assert command[command.index("-max_interleave_delta") + 1] == "100000"
    assert "-fec" not in command
    assert "-packet_loss" not in command


def test_macos_webrtc_uses_higher_quality_opus_without_fec(
    monkeypatch, tmp_path,
):
    _stub_streaming_executables(monkeypatch, tmp_path)
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="WebRTC",
        port=1122,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Darwin",
        stereo_mix_device="soundcard:Virtual Desktop Speakers",
    )
    monkeypatch.setattr(
        direct_sbs,
        "_list_darwin_audio_devices",
        lambda _ffmpeg: [(9, "Virtual Desktop Speakers")],
    )
    command = output._ffmpeg_command(1920, 1080)

    assert command[command.index("-application") + 1] == "audio"
    assert command[command.index("-b:a") + 1] == "128k"
    assert "-packet_loss" not in command
    assert "-fec" not in command
    assert "aac" not in command


def test_audio_delay_hot_update_restarts_only_ffmpeg(monkeypatch):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        stereo_mix_device="virtual-audio-capturer",
        audio_delay=-0.15,
    )
    ffmpeg_process = object()
    server_process = object()
    stopped = []
    started = []
    output.ffmpeg_process = ffmpeg_process
    output.server_process = server_process
    output._frame_size = (2, 1)
    monkeypatch.setattr(output, "_stop_process", stopped.append)
    monkeypatch.setattr(output, "_start_ffmpeg", lambda width, height: started.append((width, height)))
    monkeypatch.setattr(output, "_write_frame", lambda _frame: None)

    assert output.request_audio_delay(0.25) is True
    output.submit_frame(np.zeros((1, 2, 3), dtype=np.uint8))

    assert output.audio_delay == 0.25
    assert stopped == [ffmpeg_process]
    assert started == [(2, 1)]
    assert output.server_process is server_process


def test_nvenc_selection_falls_back_when_probe_fails(monkeypatch, capsys):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
    )
    monkeypatch.setattr(output, "_probe_nvenc", lambda width, height: False)
    monkeypatch.setattr(output, "_probe_encoder", lambda *_args: False)

    assert output._select_video_encoder(3840, 2160) == "libx264"
    assert "falling back to libx264" in capsys.readouterr().out


def test_nvenc_probe_uses_actual_sbs_resolution(monkeypatch):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(direct_sbs.subprocess, "run", fake_run)
    monkeypatch.setattr(
        direct_sbs,
        "_load_pynvvideo_codec",
        lambda: (_ for _ in ()).throw(
            AssertionError("FFmpeg NVENC probing must not load PyNvVideoCodec")
        ),
    )

    assert output._probe_nvenc(5120, 1440)
    assert "color=c=black:s=5120x1440:r=1" in commands[0]


def test_nvenc_probe_logs_ffmpeg_failure_detail(monkeypatch, capsys):
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTMP",
        port=1935,
        stream_key="live",
        fps=60,
        crf=20,
        os_name="Windows",
        prefer_nvenc=True,
    )
    monkeypatch.setattr(
        direct_sbs.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr=b"[h264_nvenc] Width 7680 exceeds encoder limit\n",
        ),
    )

    assert output._probe_nvenc(7680, 2160) is False
    output_text = capsys.readouterr().out
    assert "NVENC probe failed for 7680x2160" in output_text
    assert "Width 7680 exceeds encoder limit" in output_text


def test_mediamtx_hls_segment_limit_supports_high_bitrate_full_sbs():
    config = (APP_ROOT / "streaming/rtmp/mediamtx/mediamtx.yml").read_text(
        encoding="utf-8"
    )

    assert "hlsSegmentMaxSize: 256M" in config


def test_mediamtx_webrtc_prefers_udp_with_tcp_fallback_and_burst_queue():
    active = (APP_ROOT / "streaming/rtmp/mediamtx.yml").read_text(encoding="utf-8")
    bundled = (APP_ROOT / "streaming/rtmp/mediamtx/mediamtx.yml").read_text(
        encoding="utf-8"
    )

    assert active == bundled
    assert "writeQueueSize: 2048" in active
    assert "webrtcLocalUDPAddress: :8189" in active
    assert "webrtcLocalTCPAddress: :8189" in active


def test_rtsp_selection_uses_selected_port_for_internal_publish():
    output = FfmpegDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="RTSP",
        port=9554,
        stream_key="live",
        fps=30,
        crf=20,
        os_name="Windows",
    )

    assert output.publish_rtsp_port == 9554
    assert output._server_environment()["MTX_RTSPADDRESS"] == ":9554"


@pytest.mark.parametrize("cuda_stream", [0, 12345])
def test_pynv_encoder_uses_live_low_latency_nvenc_settings(cuda_stream):
    captured = {}

    class FakeNvc:
        @staticmethod
        def CreateEncoder(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace()

    encoder = nvidia_encoder.PyNvVideoCodecEncoder(
        FakeNvc(),
        3840,
        2160,
        hevc=False,
        fps=25,
        bitrate=21_000_000,
        cuda_stream=cuda_stream,
    )

    assert captured["args"] == (3840, 2160, "NV12", False)
    assert captured["kwargs"]["codec"] == "h264"
    assert captured["kwargs"]["tuning_info"] == "ultra_low_latency"
    assert captured["kwargs"]["preset"] == "P1"
    assert captured["kwargs"]["rc"] == "cbr"
    assert captured["kwargs"]["gop"] == 25
    assert captured["kwargs"]["idrperiod"] == 25
    assert captured["kwargs"]["bf"] == 0
    assert captured["kwargs"]["repeatspspps"] == 1
    assert captured["kwargs"]["gpu_id"] == 0
    assert captured["kwargs"]["cudastream"] == encoder._cuda_stream == cuda_stream


def test_native_timestamped_packets_are_wrapped_as_mpegts_with_pts():
    muxer = H264MpegTsTimestampMuxer(fps=25)
    packet = SimpleNamespace(data=b"\x00\x00\x00\x01\x65" + b"x" * 300, pts=1, dts=1, duration=1)

    output = muxer.wrap(packet)

    assert output
    assert len(output) % 188 == 0
    assert output[0] == 0x47
    assert output[188] == 0x47
    assert b"\x00\x00\x01\xe0" in output
    assert len(muxer.wrap(packet)) % 188 == 0


def test_pynv_muxer_copies_video_and_encodes_soundcard_pcm_as_opus(monkeypatch):
    captured = {}

    class RunningProcess:
        stdin = object()
        returncode = None

        @staticmethod
        def poll():
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return RunningProcess()

    monkeypatch.setattr(nvidia_encoder.subprocess, "Popen", fake_popen)
    output = PyNvSrtVideoOutput(
        SimpleNamespace(),
        "ffmpeg",
        codec="h264",
        fps=25,
        audio_url="udp://127.0.0.1:54321",
        audio_delay=-0.1,
        audio_codec="libopus",
        output_args=[
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            "rtsp://127.0.0.1:8554/live",
        ],
    )

    command = output.command
    assert command[command.index("-r") + 1] == "25"
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-c:a") + 1] == "libopus"
    assert command[command.index("-max_interleave_delta") + 1] == "100000"
    assert command[command.index("-thread_queue_size") + 1] == "1024"
    assert command[command.index("-probesize") + 1] == "64"
    assert command[command.index("-use_wallclock_as_timestamps") + 1] == "1"
    assert command[command.index("-muxdelay") + 1] == "0"
    assert command[command.index("-fflags") + 1] == "+nobuffer+genpts"
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert "udp://127.0.0.1:54321" in command
    assert command[-1] == "rtsp://127.0.0.1:8554/live"
    assert captured["kwargs"]["creationflags"] == 0


def test_pynv_output_uses_rtsp_for_webrtc_and_soundcard_loopback(monkeypatch):
    events = []

    class FakeSoundcardSender:
        ffmpeg_url = "udp://127.0.0.1:54321"

        def __init__(self, device_name):
            events.append(("init", device_name))

        def start(self):
            events.append(("start", None))

        def close(self):
            events.append(("close", None))

    monkeypatch.setattr(direct_sbs, "SoundcardLoopbackSender", FakeSoundcardSender)
    output = PyNvDirectSbsOutput(
        base_dir=str(APP_ROOT),
        protocol="WebRTC",
        port=1122,
        stream_key="live",
        fps=25,
        crf=23,
        os_name="Windows",
        prefer_nvenc=True,
        stereo_mix_device="soundcard:virtual-audio-capturer",
    )

    assert output._start_pynv_audio() == "udp://127.0.0.1:54321"
    assert output._pynv_output_args() == [
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        "-pkt_size",
        "1452",
        "rtsp://127.0.0.1:8554/live?pkt_size=1452",
    ]
    output._release_pynv_pipeline()
    assert events == [
        ("init", "virtual-audio-capturer"),
        ("start", None),
        ("close", None),
    ]


def test_darwin_loopback_routing_hint_only_for_virtual_devices() -> None:
    assert "BlackHole" in direct_sbs._darwin_loopback_routing_hint(":1 BlackHole 2ch")
    assert "BlackHole" in direct_sbs._darwin_loopback_routing_hint(":8 Virtual Desktop Speakers")
    assert direct_sbs._darwin_loopback_routing_hint(":2 iMac Microphone") == ""
    assert direct_sbs._darwin_loopback_routing_hint("") == ""


def test_gpu_required_stream_rejects_host_rgb_encoder_input():
    output = object.__new__(FfmpegDirectSbsOutput)
    output.gpu_required = True

    with pytest.raises(RuntimeError, match="cannot accept host RGB"):
        output.submit_frame(np.zeros((8, 16, 3), dtype=np.uint8))


def test_gpu_required_mjpeg_is_rejected_without_a_gpu_jpeg_encoder():
    with pytest.raises(RuntimeError, match="CPU JPEG encoding"):
        MjpegDirectSbsOutput(port=1122, fps=60, quality=90, gpu_required=True)


@pytest.mark.parametrize("display_mode", ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB"])
@pytest.mark.parametrize("fit_mode", ["contain", "cover", "stretch"])
@pytest.mark.parametrize("percent", ["100%", "85%", "75%", "50%"])
def test_quality_scale_uses_scaled_transport_without_reupscale(
    display_mode, fit_mode, percent
):
    from stereo_runtime.render_size import RenderSizeConfig, resolve_render_size

    original = (160, 100)
    source = resolve_render_size(original, RenderSizeConfig(scale_factor=percent))
    config = SimpleNamespace(output_quality_enabled=False, output_format=display_mode.lower().replace("-", "_"))
    target = direct_sbs.stream_target_size_for_config(source, config, fit_mode)
    unscaled_target = direct_sbs.stream_target_size_for_config(original, config, fit_mode)
    if percent == "100%":
        assert target == unscaled_target
    else:
        assert target[0] <= unscaled_target[0]
        assert target[1] <= unscaled_target[1]
    width = source[0] * (2 if display_mode == "Full-SBS" else 1)
    height = source[1] * (2 if display_mode == "Full-TAB" else 1)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    if display_mode.endswith("SBS"):
        pixels[:, :width // 2, 0] = 255
        pixels[:, width // 2:, 1] = 255
    else:
        pixels[:height // 2, :, 0] = 255
        pixels[height // 2:, :, 1] = 255
    converter = RuntimeSbsRgbConverter(
        display_mode=display_mode, fit_mode=fit_mode, input_size=original, target_size=target,
    )
    result = converter.convert(pixels)
    assert result.shape[:2] == (target[1], target[0])
    if display_mode.endswith("SBS"):
        left, right = result[:, :target[0] // 2], result[:, target[0] // 2:]
    else:
        left, right = result[:target[1] // 2], result[target[1] // 2:]
    assert left[..., 0].max() == 255 and left[..., 1].max() == 0
    assert right[..., 1].max() == 255 and right[..., 0].max() == 0


@pytest.mark.parametrize("source_size", [(1920, 1080), (3840, 2160), (1080, 1920)])
def test_stream_target_preserves_existing_output_quality(source_size, monkeypatch):
    from stereo_runtime.output_quality import output_sampling_plan_for_config
    from streaming.aspect import transport_canvas_size

    monkeypatch.delenv("D2S_CAP_OUTPUT_UPSCALE", raising=False)
    config = SimpleNamespace(output_quality_enabled=True, output_headset_tier_k=4, output_format="half_sbs")
    plan = output_sampling_plan_for_config(config, *source_size)
    expected = transport_canvas_size(
        (plan.target_width, plan.target_height), "contain", input_size=source_size, display_mode="Half-SBS",
    )
    assert direct_sbs.stream_target_size_for_config(source_size, config, "contain") == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("display_mode", ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB"])
@pytest.mark.parametrize("fit_mode", ["contain", "cover", "stretch"])
def test_scaled_gpu_consumer_restores_encoder_canvas_without_host_readback(monkeypatch, display_mode, fit_mode):
    shutdown = threading.Event()
    submitted = []
    width, height = (160, 50) if display_mode == "Full-SBS" else (80, 100) if display_mode == "Full-TAB" else (80, 50)
    image = torch.zeros((height, width, 3), device="cuda", dtype=torch.uint8)
    if display_mode.endswith("SBS"):
        image[:, :width // 2, 0] = 255
        image[:, width // 2:, 1] = 255
    else:
        image[:height // 2, :, 0] = 255
        image[height // 2:, :, 1] = 255
    config = SimpleNamespace(output_quality_enabled=False, output_format=display_mode.lower().replace("-", "_"))
    target = direct_sbs.stream_target_size_for_config((160, 100), config, fit_mode)
    output = SimpleNamespace(display_mode=display_mode, fit_mode=fit_mode, input_size=(160, 100))

    def submit(frame):
        assert frame.is_cuda
        submitted.append(frame)
        shutdown.set()

    output.submit_cuda_frame = submit
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *_a, **_k: pytest.fail("GPU frame was read back"))
    consumer = DirectSbsOutputConsumer(
        runtime_q=queue.Queue(), shutdown_event=shutdown, output=output,
        source_stat_inc=lambda *_a, **_k: None, gpu_required=True, target_size=target,
    )
    consumer.runtime_q.put((SimpleNamespace(sbs=image), 0.0))
    consumer.run()
    assert len(submitted) == 1
    assert tuple(submitted[0].shape) == (target[1], target[0], 3)


def test_native_consumer_receives_target_canvas_before_submission():
    shutdown = threading.Event()
    seen = []

    def submit(result, *, target_size):
        seen.append((result.vulkan_compute_request, target_size))
        shutdown.set()
        return True

    output = SimpleNamespace(submit_vulkan_stereo_frame=submit, fit_mode="stretch", display_mode="Half-SBS")
    consumer = DirectSbsOutputConsumer(
        runtime_q=queue.Queue(), shutdown_event=shutdown, output=output,
        source_stat_inc=lambda *_a, **_k: None, target_size=(1920, 1080), gpu_required=True,
    )
    request = object()
    consumer.runtime_q.put((SimpleNamespace(vulkan_compute_request=request), 0.0))
    consumer.run()
    assert seen == [(request, (1920, 1080))]


def test_mjpeg_target_canvas_is_owned_by_existing_resize_profile(monkeypatch):
    captured = []

    def streamer(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(direct_sbs, "MJPEGStreamer", streamer)
    MjpegDirectSbsOutput(port=1122, fps=60, quality=90, target_size=(1920, 1080))
    assert captured[0]["profile"].resize_size == (1920, 1080)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("layout", ["hwc", "chw", "bchw", "bhwc", "strided_rgba"])
def test_nv12_fused_conversion_matches_existing_gpu_bytes(monkeypatch, dtype, layout):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVENC conversion requires a CUDA GPU, not ROCm")
    from stereo_runtime import triton_runtime

    if not triton_runtime.triton_runtime_available(torch.device("cuda")):
        pytest.skip("optional CUDA Triton runtime is unavailable")
    generator = torch.Generator(device="cuda").manual_seed(731)
    image = torch.randint(0, 256, (1080, 1920, 4), device="cuda", dtype=torch.uint8, generator=generator)
    if dtype == torch.float32:
        image = image.float() / 255.0 * 1.4 - 0.2
    if layout == "strided_rgba":
        image = image[:, ::2, :]
    else:
        image = image[..., :3]
        if layout in {"chw", "bchw"}:
            image = image.permute(2, 0, 1).contiguous()
        if layout in {"bchw", "bhwc"}:
            image = image.unsqueeze(0)
    fused = nvidia_encoder.rgb_cuda_to_nv12(image)
    monkeypatch.setattr(triton_runtime, "triton_runtime_available", lambda _device: False)
    reference = nvidia_encoder.rgb_cuda_to_nv12(image)
    assert all(torch.equal(actual, expected) for actual, expected in zip(fused, reference))
    assert fused[0].untyped_storage().data_ptr() == fused[1].untyped_storage().data_ptr()
    assert fused[0].dtype == fused[1].dtype == torch.uint8
    assert fused[0].is_cuda and fused[1].is_cuda


@pytest.mark.parametrize("failure", [ImportError, RuntimeError])
def test_nv12_optional_kernel_failure_keeps_existing_conversion(monkeypatch, failure):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVENC conversion requires a CUDA GPU, not ROCm")
    from stereo_runtime import triton_runtime

    image = torch.linspace(-0.25, 1.25, 32 * 64 * 3, device="cuda").reshape(32, 64, 3)
    monkeypatch.setattr(triton_runtime, "triton_runtime_available", lambda _device: False)
    reference = nvidia_encoder.rgb_cuda_to_nv12(image)
    monkeypatch.setattr(triton_runtime, "triton_runtime_available", lambda _device: True)

    def unavailable():
        raise failure("optional compiler unavailable")

    monkeypatch.setattr(nvidia_encoder, "_nv12_kernel", unavailable)
    fallback = nvidia_encoder.rgb_cuda_to_nv12(image)
    assert all(torch.equal(actual, expected) for actual, expected in zip(fallback, reference))


def test_nv12_rejects_host_frame_without_a_readback():
    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        nvidia_encoder.rgb_cuda_to_nv12(torch.zeros(32, 64, 3))


@pytest.mark.parametrize("handoff", ["shared", "different", "default"])
@pytest.mark.parametrize("fused", [False, True])
def test_nv12_waits_only_when_encoder_stream_is_not_shared(monkeypatch, handoff, fused):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVENC stream hand-off requires a CUDA GPU, not ROCm")
    from stereo_runtime import triton_runtime

    if fused and not triton_runtime.triton_runtime_available(torch.device("cuda")):
        pytest.skip("optional CUDA Triton runtime is unavailable")
    if not fused:
        monkeypatch.setattr(triton_runtime, "triton_runtime_available", lambda _device: False)
    producer = torch.cuda.Stream()
    other = torch.cuda.Stream()
    original_sync = torch.cuda.Stream.synchronize
    waits = []

    def synchronize(stream):
        waits.append(stream.cuda_stream)
        original_sync(stream)

    with torch.cuda.stream(producer):
        image = torch.linspace(0, 1, 32 * 64 * 3, device="cuda").reshape(32, 64, 3)
        reference = nvidia_encoder.rgb_cuda_to_nv12(image)
        monkeypatch.setattr(torch.cuda.Stream, "synchronize", synchronize)
        stream_handle = (
            producer.cuda_stream if handoff == "shared" else other.cuda_stream if handoff == "different" else 0
        )
        planes = nvidia_encoder.rgb_cuda_to_nv12(image, encoder_stream=stream_handle)
    assert waits == ([] if handoff == "shared" else [producer.cuda_stream])
    original_sync(producer)
    assert all(torch.equal(actual, expected) for actual, expected in zip(planes, reference))
    assert planes[0].is_cuda and planes[1].is_cuda


@pytest.mark.parametrize("encoder_kind", ["pynv", "direct_nv12"])
@pytest.mark.parametrize("hevc", [False, True])
@pytest.mark.parametrize("vertical", [False, True])
def test_shared_stream_nvenc_decode_preserves_eye_colors_and_frame_order(tmp_path, hevc, vertical, encoder_kind):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVENC/NVDEC hardware round-trip requires a CUDA GPU, not ROCm")
    nvc = direct_sbs._load_pynvvideo_codec()
    if nvc is None:
        pytest.skip("optional PyNvVideoCodec runtime is unavailable")
    stream = torch.cuda.Stream()
    width, height, count = 256, 144, 12
    with torch.cuda.stream(stream):
        if encoder_kind == "direct_nv12":
            from streaming.nvenc_cudaarray_bridge import NvencCudaNv12Encoder

            try:
                encoder = NvencCudaNv12Encoder(width, height, hevc=hevc, fps=60, bitrate=8_000_000, device="cuda:0")
            except (AttributeError, OSError, RuntimeError) as exc:
                pytest.skip(f"native NV12 bridge is unavailable: {exc}")
        else:
            encoder = nvidia_encoder.PyNvVideoCodecEncoder(
                nvc, width, height, hevc=hevc, fps=60, bitrate=8_000_000,
                cuda_stream=stream.cuda_stream,
            )
        packets = []
        for index in range(count):
            image = torch.zeros((height, width, 3), device="cuda", dtype=torch.uint8)
            if vertical:
                image[:height // 2, :, index % 2] = 255
                image[height // 2:, :, 1 - index % 2] = 255
            else:
                image[:, :width // 2, index % 2] = 255
                image[:, width // 2:, 1 - index % 2] = 255
            emitted = encoder.encode_access_units(image, frame_index=index, force_idr=index in {0, 6})
            if encoder_kind == "direct_nv12":
                assert [packet["frame_index"] for packet in emitted] == [index], "encoder buffered a frame"
                if index in {0, 6}:
                    from streaming.moonlight_output import MoonlightSbsOutput

                    assert MoonlightSbsOutput._is_idr(emitted[0]["data"], "hevc" if hevc else "h264")
            packets.extend(emitted)
        packets.extend(encoder.flush_access_units())
        close = getattr(encoder, "close", None)
        if callable(close):
            close()
            close()  # Resource cleanup remains idempotent.
    assert [packet["frame_index"] for packet in packets] == list(range(count))
    bitstream = tmp_path / ("stereo.hevc" if hevc else "stereo.h264")
    bitstream.write_bytes(b"".join(packet["data"] for packet in packets))
    decoder = nvc.CreateDecoder(
        gpuid=0, codec=nvc.cudaVideoCodec.HEVC if hevc else nvc.cudaVideoCodec.H264,
        cudastream=stream.cuda_stream, usedevicememory=True,
        outputColorType=nvc.OutputColorType.RGBP, latency=nvc.DisplayDecodeLatencyType.ZERO,
    )
    decoded_count = 0
    with torch.cuda.stream(stream):
        for packet in nvc.CreateDemuxer(str(bitstream)):
            for frame in decoder.Decode(packet):
                rgb = torch.from_dlpack(frame)
                assert rgb.is_cuda and tuple(rgb.shape) == (3, height, width)
                first = rgb[:, height // 4, width // 2] if vertical else rgb[:, height // 2, width // 4]
                second = rgb[:, 3 * height // 4, width // 2] if vertical else rgb[:, height // 2, 3 * width // 4]
                channel = decoded_count % 2
                assert first[channel].item() > 180 and first[1 - channel].item() < 60
                assert second[1 - channel].item() > 180 and second[channel].item() < 60
                decoded_count += 1
    assert decoded_count == count


@pytest.mark.parametrize("fused", [True, False])
def test_nv12_registered_destination_reuses_exact_existing_gpu_planes(monkeypatch, fused):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVENC NV12 conversion requires NVIDIA CUDA")
    from stereo_runtime import triton_runtime

    if not fused:
        monkeypatch.setattr(triton_runtime, "triton_runtime_available", lambda _device: False)
    image = torch.linspace(-0.1, 1.1, 32 * 64 * 3, device="cuda").reshape(32, 64, 3)
    expected = nvidia_encoder.rgb_cuda_to_nv12(image)
    target = torch.empty((48, 64, 1), dtype=torch.uint8, device="cuda")
    planes = nvidia_encoder.rgb_cuda_to_nv12(image, destination=target)
    assert all(plane.untyped_storage().data_ptr() == target.data_ptr() for plane in planes)
    assert all(torch.equal(actual, reference) for actual, reference in zip(planes, expected))
    for invalid in (target.float(), target[:32], target[:, ::2], target.transpose(0, 1)):
        with pytest.raises(ValueError, match="NV12 destination"):
            nvidia_encoder.rgb_cuda_to_nv12(image, destination=invalid)
