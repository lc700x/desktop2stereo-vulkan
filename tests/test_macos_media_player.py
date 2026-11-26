from __future__ import annotations

import os
import sys
import time
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation is macOS-only")
def test_avfoundation_native_output_yields_retained_gpu_frames():
    media_path = os.environ.get("D2S_TEST_MEDIA_PATH")
    if not media_path:
        pytest.skip("set D2S_TEST_MEDIA_PATH to an HDR test video")

    from capture.native.macos_media_player import MacOSMediaPlayer

    try:
        player = MacOSMediaPlayer(media_path, hdr=True, audio_enabled=False)
    except RuntimeError as exc:
        if "hardware decoder" in str(exc).lower() or "cannot decode" in str(exc).lower():
            pytest.skip(f"hardware decoder unavailable for test media: {exc}")
        raise
    frames = []
    deadline = time.monotonic() + 4.0
    try:
        while time.monotonic() < deadline:
            try:
                frame = player.copy_frame()
            except RuntimeError as exc:
                if "hardware decoder" in str(exc).lower() or "cannot decode" in str(exc).lower():
                    pytest.skip(f"hardware decoder unavailable for test media: {exc}")
                raise
            if frame is None:
                time.sleep(0.005)
                continue
            frames.append((frame.presentation_time, frame.resource_format))
            frame.release()
    finally:
        player.close()

    assert len(frames) >= 2, "VideoToolbox returned no usable GPU video frames"
    assert all(pixel_format == "P010_10Bit" for _, pixel_format in frames)
    assert all(after > before for (before, _), (after, _) in pairwise(frames))


@pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation is macOS-only")
def test_mkv_remux_keeps_video_on_container_copy_path(monkeypatch, tmp_path):
    import capture.native.macos_media_player as media_player

    source = tmp_path / "sample.mkv"
    source.write_bytes(b"matroska fixture")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"mp4 fixture")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(media_player, "_ffmpeg_binary", lambda: "/test/ffmpeg")
    monkeypatch.setattr(media_player.subprocess, "run", fake_run)

    remuxed, temporary_dir = media_player._prepare_media_asset(source)
    try:
        assert remuxed.endswith(".mp4")
        assert temporary_dir is not None
        command, kwargs = calls[0]
        assert "-c" in command and command[command.index("-c") + 1] == "copy"
        assert "-map" in command and "0:v:0" in command
        assert kwargs["timeout"] == 120
    finally:
        media_player._cleanup_media_asset(temporary_dir)
