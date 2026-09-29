from __future__ import annotations

from pathlib import Path

from streaming.direct_sbs import VulkanDirectSbsOutput


def test_vulkan_filter_options_are_before_output_url(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("D2S_STREAMING_RUNTIME_DIR", str(tmp_path))
    output = object.__new__(VulkanDirectSbsOutput)
    output.video_encoder = "h264_vulkan"
    output.use_hevc = False
    output.crf = 23
    output._ffmpeg_command = VulkanDirectSbsOutput._ffmpeg_command.__get__(output)

    # Exercise only the ordering transformation with a representative base
    # command; no MediaMTX or FFmpeg process is started by this test.
    base = ["ffmpeg", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "rtsp", "rtsp://127.0.0.1/live"]
    monkeypatch.setattr(
        "streaming.direct_sbs.FfmpegDirectSbsOutput._ffmpeg_command",
        lambda _self, _width, _height: list(base),
    )
    command = output._ffmpeg_command(3840, 2160)
    assert command[-1].startswith("rtsp://")
    assert command.index("-vf") < len(command) - 1
    assert command[command.index("-vf") + 1] == "format=nv12,hwupload"


def test_vulkan_calibration_uses_shared_ffmpeg_command(monkeypatch) -> None:
    output = object.__new__(VulkanDirectSbsOutput)
    output._calibration_controller = object()
    base = ["ffmpeg", "-f", "lavfi", "-i", "testsrc2", "rtsp://127.0.0.1/live"]
    monkeypatch.setattr(
        "streaming.direct_sbs.FfmpegDirectSbsOutput._ffmpeg_command",
        lambda _self, _width, _height: list(base),
    )

    command = VulkanDirectSbsOutput._ffmpeg_command(output, 3840, 2160)

    assert command == base
    assert "-init_hw_device" not in command
    assert "-vf" not in command


def test_native_vulkan_publish_url_uses_local_rtsp_packet_size() -> None:
    output = object.__new__(VulkanDirectSbsOutput)
    output.protocol = "WEBRTC"
    output.port = 1122
    output.stream_key = "live"
    assert output._native_output_url() == "rtsp://127.0.0.1:8554/live?pkt_size=1452"

