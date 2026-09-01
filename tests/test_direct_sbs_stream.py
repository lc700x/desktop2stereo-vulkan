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
    PyNvDirectSbsOutput,
    RuntimeSbsRgbConverter,
    VulkanDirectSbsOutput,
    runtime_sbs_to_rgb,
)
from streaming.nvidia_encoder import H264MpegTsTimestampMuxer, PyNvSrtVideoOutput


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

    # Empty / bare-prefix / lone-colon / "no device" values must not reach
    # FFmpeg as "-i :" (it aborts with "Error opening input file :."); on
    # macOS they auto-select a loopback device so the stream carries sound.
    for bad in ("", "soundcard:", "wasapi:", ":", "No Stereo Mix device found"):
        target.stereo_mix_device = bad
        args = target._audio_input_args()
        assert args, f"device {bad!r} must auto-select audio"
        assert args[-1].startswith(":")


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
    converter = RuntimeSbsRgbConverter()
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


def test_macos_rtmp_stream_audio_uses_opus_like_v250(monkeypatch) -> None:
    # v2.5.0 macOS parity: every non-WebRTC client protocol (RTMP/RTSP/HLS)
    # publishes Opus 96k/48k to MediaMTX's RTSP ingest; AAC is only used on
    # the Windows/Linux SRT/RTMP paths (NVIDIA/ROCm soundcard chain).
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
    assert command[command.index("-af") + 1] == "aresample=async=1"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"
    assert command[command.index("-b:a") + 1] == "96k"
    assert command[command.index("-rtbufsize") + 1] == "256M"
    assert "aac" not in command
    # The device NAME is resolved to its AVFoundation index; the audio
    # input's "-i" follows "-rtbufsize" (the video input is rawvideo pipe:0).
    audio_i = command.index("-i", command.index("-rtbufsize"))
    assert command[audio_i + 1] == ":8"
    # Professional A/V sync: both inputs share the av_gettime() wall-clock
    # base (video PTS can never drift behind the real-time audio clock) and
    # each input demuxes on its own thread (audio can't be starved by a
    # stalled video pipe producer).
    assert command.count("-use_wallclock_as_timestamps") == 2
    assert command.count("-thread_queue_size") == 2
    assert command[command.index("-use_wallclock_as_timestamps") + 1] == "1"


def test_macos_webrtc_stream_audio_uses_opus_async1(monkeypatch) -> None:
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
    assert command[command.index("-af") + 1] == "aresample=async=1"
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


def test_pynv_encoder_uses_live_low_latency_nvenc_settings():
    captured = {}

    class FakeNvc:
        @staticmethod
        def CreateEncoder(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace()

    nvidia_encoder.PyNvVideoCodecEncoder(
        FakeNvc(),
        3840,
        2160,
        hevc=False,
        fps=25,
        bitrate=21_000_000,
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
