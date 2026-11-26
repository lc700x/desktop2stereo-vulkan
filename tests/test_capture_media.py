import io
import json
import os
import sys
import subprocess
import tempfile
import threading
import time
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from capture.media import (
    MediaCaptureSource,
    _amf_vpp_filter,
    _is_amf_vpp_unavailable,
    _is_hardware_decode_unavailable,
    _is_software_decoder_unavailable,
    _media_frame_format,
    _preferred_audio_output_device,
    _probe_media,
    _resolve_amf_output_size,
    _resolve_media_playback_cadence,
    _select_media_cadence_fps_for_display,
    _video_filter,
    _video_pixel_format,
    get_media_playback_cadence_target,
    reset_media_playback_cadence_target,
    set_media_playback_cadence_target,
)


class PyNvVCExceptionUnsupported(RuntimeError):
    pass


def _source_for_decode_start():
    source = object.__new__(MediaCaptureSource)
    source._video_process = None
    source._nvdec_source_path = "sample.mp4"
    source._nvdec_decoder = None
    source._nvdec_torch = None
    source._nvdec_disabled = False
    source._codec_name = "h264"
    source._amf_decoder = None
    source._amf_disabled = False
    source._nvdec_first_frame_pending = False
    source._nvdec_next_frame_at = None
    source._decode_backend = "ffmpeg-software"
    source._frame_format = "rgbp"
    source._frame_pixel_format = "bgr24"
    source._gpu_vendor = "nvidia"
    source.hdr = False
    source.transfer = ""
    source.primaries = ""
    source.decode_fallback_reason = ""
    source._start_ffmpeg_video = lambda position, *, realtime: None
    return source


def test_media_runtime_metrics_report_pipe_read_and_buffer_pressure():
    source = object.__new__(MediaCaptureSource)
    source._frame_pump_metrics_lock = threading.Lock()
    source._frame_pump_read_ms = deque((10.0, 20.0, 30.0, 40.0, 50.0))
    source._frame_pump_frame_times = deque((1.0, 2.0, 3.0, 4.0, 5.0))
    source._frame_buffer_condition = threading.Condition()
    source._frame_buffer = deque((1, 2, 3))
    source._frame_buffer_capacity = 8
    source._frame_pump_done = False
    source._decode_backend = "ffmpeg-amf"
    source.decode_fallback_reason = ""

    metrics = source.runtime_metrics()

    assert metrics["decode_fps"] == pytest.approx(1.0)
    assert metrics["read_ms_p50"] == pytest.approx(30.0)
    assert metrics["read_ms_p95"] == pytest.approx(50.0)
    assert metrics["read_ms_max"] == pytest.approx(50.0)
    assert metrics["buffer_depth"] == 3
    assert metrics["buffer_capacity"] == 8
    assert metrics["decode_backend"] == "ffmpeg-amf"


def test_media_stop_serializes_concurrent_audio_cleanup():
    source = object.__new__(MediaCaptureSource)
    source._media_control_lock = threading.RLock()
    source._current_position = lambda: 0.0
    source._video_process = None
    source._displayed_frame = None
    source._manifest_path = None
    source._owns_iso_mount = False
    source._mounted_iso_path = None
    source.playing = True
    source._ready = True
    source._browser = SimpleNamespace(close=lambda: None)
    source._stop_frame_pump = lambda: None
    source._close_native_decoder = lambda: None
    source._close_nvdec = lambda: None
    source._close_opencv = lambda: None
    source._reset_frame_buffer = lambda _position: None
    source._publish_state = lambda **_kwargs: None

    entered = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def close_audio():
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        entered.set()
        release.wait(timeout=1.0)
        with state_lock:
            active -= 1

    source._close_audio = close_audio

    workers = [threading.Thread(target=source.stop) for _ in range(2)]
    for worker in workers:
        worker.start()
    assert entered.wait(timeout=1.0)
    time.sleep(0.02)
    release.set()
    for worker in workers:
        worker.join(timeout=1.0)

    assert maximum_active == 1


def test_polling_runner_stop_signals_media_control_worker_before_source_stop():
    from capture.runners import PollingCaptureRunner

    events = []
    source = SimpleNamespace(
        mode="video",
        stop=lambda: events.append("source_stop"),
    )
    runner = PollingCaptureRunner(SimpleNamespace(), lambda: source)
    runner._source = source
    runner._source_stopped = False
    control_stop = threading.Event()
    runner._control_stop = control_stop

    runner.stop()

    assert control_stop.is_set()
    assert events == ["source_stop"]


def test_media_audio_cleanup_waits_for_audio_worker_before_portaudio(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    source = object.__new__(MediaCaptureSource)
    source._audio_stop = threading.Event()
    events = []

    class Process:
        stdout = None
        _media_stderr = None

        @staticmethod
        def poll():
            return None

        def terminate(self):
            events.append("process_terminate")

        def wait(self, timeout=None):
            events.append("process_wait")
            return 0

    class Stream:
        def abort(self):
            events.append("stream_abort")

        def stop(self):
            events.append("stream_stop")

        def close(self):
            events.append("stream_close")

    source._audio_process = Process()
    source._audio_stream = Stream()
    source._audio_sync = SimpleNamespace(close=lambda: events.append("sync_close"))
    source._audio_thread = None

    source._close_audio()

    assert events.index("process_terminate") < events.index("stream_stop")
    assert "process_wait" not in events


def test_media_audio_worker_owns_portaudio_teardown(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    events = []

    class Pipe:
        def read(self, _count):
            return b""

    class Process:
        stdout = Pipe()

        @staticmethod
        def poll():
            return 0

    class Stream:
        def stop(self):
            events.append("stream_stop")

        def close(self):
            events.append("stream_close")

    source = object.__new__(MediaCaptureSource)
    source._audio_process = Process()
    source._audio_stream = Stream()
    source._audio_stop = threading.Event()
    source._audio_sync = SimpleNamespace()
    source.volume, source.muted = 1.0, False

    source._audio_loop()

    assert events == ["stream_close"]


def test_ffmpeg_decoder_keeps_realtime_pacing_after_nvdec_is_disabled(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source._video_process = None
    source._nvdec_disabled = True
    source._ffmpeg = "ffmpeg"
    source._input_args = ["-i", "sample.mp4"]
    source._video_filter = "fps=30,format=p010le"
    source._amf_video_filter = (
        "vpp_amf=w=1920:h=960:format=p010le,hwdownload,format=p010le,fps=30"
    )
    source._amf_output_size = (1920, 960)
    source._frame_pixel_format = "p010le"
    source.width = 3840
    source.height = 1920
    command = []
    process = SimpleNamespace()

    def fake_popen(args, **_kwargs):
        command.extend(args)
        return process

    monkeypatch.setattr("capture.media.subprocess.Popen", fake_popen)

    try:
        source._start_ffmpeg_video(0.0, realtime=True, decoder="h264_amf")

        assert "-re" in command
        assert command[command.index("-c:v") + 1] == "h264_amf"
        assert command[command.index("-vf") + 1] == source._amf_video_filter
        assert (source._decode_width, source._decode_height) == (1920, 960)
    finally:
        process._media_stderr.close()


def test_software_decode_uses_original_dimensions_after_amf_vpp_fallback(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source._video_process = None
    source._ffmpeg = "ffmpeg"
    source._input_args = ["-i", "sample.mp4"]
    source._video_filter = "fps=30,format=p010le"
    source._amf_video_filter = "vpp_amf=w=1920:h=960:format=p010le"
    source._amf_output_size = (1920, 960)
    source._frame_pixel_format = "p010le"
    source.width = 3840
    source.height = 1920
    command = []
    process = SimpleNamespace()

    def fake_popen(args, **_kwargs):
        command.extend(args)
        return process

    monkeypatch.setattr("capture.media.subprocess.Popen", fake_popen)

    try:
        source._start_ffmpeg_video(0.0, realtime=True)

        assert command[command.index("-vf") + 1] == source._video_filter
        assert (source._decode_width, source._decode_height) == (3840, 1920)
    finally:
        process._media_stderr.close()


def test_hardware_unsupported_errors_are_distinguished_from_corrupt_media():
    assert _is_hardware_decode_unavailable(PyNvVCExceptionUnsupported("unsupported codec"))
    assert _is_hardware_decode_unavailable(RuntimeError("Codec not supported on this GPU"))
    assert _is_hardware_decode_unavailable(
        RuntimeError("Your platform doesn't support hardware accelerated AV1 decoding")
    )
    assert _is_hardware_decode_unavailable(ModuleNotFoundError("PyNvVideoCodec"))
    assert _is_hardware_decode_unavailable(BufferError("Unsupported device_type: 2"))
    assert _is_software_decoder_unavailable(RuntimeError("Decoder not found for codec av1"))
    assert not _is_software_decoder_unavailable(RuntimeError("Invalid data found"))
    assert not _is_hardware_decode_unavailable(ValueError("corrupt packet data"))


def test_macos_media_decode_fails_closed_until_native_metal_path(monkeypatch):
    source = _source_for_decode_start()
    source._gpu_vendor = "apple"
    source._nvdec_disabled = True
    source._frame_pixel_format = "p010le"
    source._nvdec_decoder = None
    source._close_nvdec = lambda: None
    source._close_opencv = lambda: None
    source._close_native_decoder = lambda: None
    source._start_ffmpeg_video = lambda *_args, **_kwargs: pytest.fail(
        "macOS media must not return decoded frames through FFmpeg stdout"
    )
    monkeypatch.setattr("capture.media.sys.platform", "darwin")

    with pytest.raises(RuntimeError, match="CPU frame readback is disabled"):
        source._start_video(4.5, realtime=True)

    assert source._decode_backend == "avfoundation-metal-unavailable"


def test_amf_vpp_format_negotiation_errors_are_detected():
    assert _is_amf_vpp_unavailable(
        "[Parsed_vpp_amf_0 @ 0x1] Impossible to convert between the formats "
        "supported by the filter and Parsed_hwdownload_1"
    )
    assert _is_amf_vpp_unavailable(
        "Could not convert between the formats supported by filter vpp_amf"
    )
    assert not _is_amf_vpp_unavailable("[Parsed_vpp_amf_0] Invalid packet data")


def test_video_decode_prefers_nvdec_when_available():
    source = _source_for_decode_start()
    software_starts = []
    source._start_ffmpeg_video = lambda position, *, realtime: software_starts.append(position)

    def start_nvdec(position):
        source._decode_backend = "nvdec"

    source._start_nvdec = start_nvdec
    source._start_video(3.0, realtime=True)

    assert source._decode_backend == "nvdec"
    assert software_starts == []
    assert source._nvdec_disabled is False


def test_video_decode_uses_amf_on_amd():
    source = _source_for_decode_start()
    source._amf_decoder = "h264_amf"
    source._gpu_vendor = "amd"
    starts = []

    def start_ffmpeg(position, *, realtime, decoder=None):
        starts.append((position, realtime, decoder))
        source._decode_backend = "ffmpeg-amf" if decoder else "ffmpeg-software"

    source._start_ffmpeg_video = start_ffmpeg
    source._start_video(4.5, realtime=True)

    assert starts == [(4.5, True, "h264_amf")]
    assert source._decode_backend == "ffmpeg-amf"
    assert source._nvdec_disabled is False


def test_video_decode_falls_back_to_ffmpeg_for_unsupported_hardware(monkeypatch):
    monkeypatch.setattr("capture.media.sys.platform", "linux")
    source = _source_for_decode_start()
    software_starts = []
    source._start_ffmpeg_video = lambda position, *, realtime: software_starts.append(
        (position, realtime)
    )
    source._start_nvdec = lambda position: (_ for _ in ()).throw(
        PyNvVCExceptionUnsupported("Codec not supported on this GPU")
    )

    source._start_video(4.5, realtime=True)

    assert source._decode_backend == "ffmpeg-software"
    assert source._nvdec_disabled is True
    assert source._frame_format == "bgr24"
    assert software_starts == [(4.5, True)]
    assert "not supported" in source.decode_fallback_reason.casefold()


def test_nvdec_without_first_frame_falls_back_to_software(monkeypatch):
    monkeypatch.setattr("capture.media.sys.platform", "linux")
    source = _source_for_decode_start()
    source._amf_decoder = None
    starts = []
    source._start_ffmpeg_video = lambda position, *, realtime: starts.append(
        (position, realtime)
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                             Stream=lambda **_kwargs: SimpleNamespace(cuda_stream=0x123, synchronize=lambda: None)),
    )
    fake_decoder = SimpleNamespace(
        get_batch_frames=lambda _count: [],
        stop=lambda: None,
    )
    fake_nvcodec = SimpleNamespace(
        OutputColorType=SimpleNamespace(RGBP=object()),
        CreateSimpleDecoder=lambda **_kwargs: fake_decoder,
    )
    monkeypatch.setattr("capture.media._enable_torch_cuda_dlls", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_nvcodec)

    source._start_video(0.0, realtime=True)

    assert starts == [(0.0, True)]
    assert source._decode_backend == "ffmpeg-software"
    assert source._nvdec_disabled is True
    assert "could not decode the first video frame" in source.decode_fallback_reason.casefold()
    assert source._nvdec_decoder is None


def test_nvdec_prefers_threaded_decoder_and_maps_resume_position(monkeypatch):
    source = _source_for_decode_start()
    source._source_fps = 120.0
    source._playback_fps = 60.0
    source._decode_nvdec_frame = lambda: object()
    mapped_positions = []
    seek_indices = []
    preflight_batches = []
    disposed_seekers = []
    created_threaded = []

    class SeekDecoder:
        def get_index_from_time_in_seconds(self, position):
            mapped_positions.append(position)
            return 300

        def seek_to_index(self, frame_index):
            seek_indices.append(frame_index)

        def get_batch_frames(self, count):
            preflight_batches.append(count)
            return [object()]

        def stop(self):
            disposed_seekers.append(True)

    class FakeThreadedDecoder:
        def __init__(self, path, **kwargs):
            created_threaded.append((path, kwargs))

        def get_batch_frames(self, _count):
            return []

        def end(self):
            pass

    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                             Stream=lambda **_kwargs: SimpleNamespace(cuda_stream=0x123, synchronize=lambda: None)),
    )
    fake_nvcodec = SimpleNamespace(
        OutputColorType=SimpleNamespace(RGBP=object()),
        CreateSimpleDecoder=lambda **_kwargs: SeekDecoder(),
        ThreadedDecoder=FakeThreadedDecoder,
    )
    monkeypatch.setattr("capture.media._enable_torch_cuda_dlls", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_nvcodec)

    source._start_nvdec(5.0)

    assert mapped_positions == [5.0]
    assert seek_indices == [300]
    assert preflight_batches == [1]
    assert disposed_seekers == [True]
    assert created_threaded[0][0] == "sample.mp4"
    assert created_threaded[0][1]["start_frame"] == 300
    assert created_threaded[0][1]["buffer_size"] == 4
    assert created_threaded[0][1]["cuda_stream"] == 0x123
    assert source._decode_backend == "nvdec"


def test_nvdec_unsupported_preflight_falls_back_before_threaded_decoder(monkeypatch):
    monkeypatch.setattr("capture.media.sys.platform", "linux")
    source = _source_for_decode_start()
    starts = []
    threaded_starts = []
    source._start_ffmpeg_video = lambda position, *, realtime: starts.append(
        (position, realtime)
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                             Stream=lambda **_kwargs: SimpleNamespace(cuda_stream=0x123, synchronize=lambda: None)),
    )

    class UnsupportedDecoder:
        def get_batch_frames(self, _count):
            raise PyNvVCExceptionUnsupported("Codec not supported on this GPU")

        def stop(self):
            pass

    def create_threaded_decoder(*_args, **_kwargs):
        threaded_starts.append(True)
        raise AssertionError("ThreadedDecoder must not start before NVDEC preflight")

    fake_nvcodec = SimpleNamespace(
        OutputColorType=SimpleNamespace(RGBP=object()),
        CreateSimpleDecoder=lambda **_kwargs: UnsupportedDecoder(),
        ThreadedDecoder=create_threaded_decoder,
    )
    monkeypatch.setattr("capture.media._enable_torch_cuda_dlls", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_nvcodec)

    source._start_video(0.0, realtime=True)

    assert starts == [(0.0, True)]
    assert threaded_starts == []
    assert source._decode_backend == "ffmpeg-software"
    assert source._nvdec_disabled is True
    assert "not supported" in source.decode_fallback_reason.casefold()


def test_nvdec_native_p010_normalizes_byte_like_dlpack_stride():
    class FakeFrame:
        def __init__(self, *, strided):
            self.ndim = 2
            self.shape = (4, 6)
            self.dtype = object()
            self._stride = (6, 2) if strided else (6, 1)
            self.device = SimpleNamespace(type="cuda")
            self.cloned_stride = None

        def stride(self):
            return self._stride

        def as_strided(self, shape, stride):
            assert shape == self.shape
            assert stride == (6, 1)
            return FakeFrame(strided=False)

        def clone(self):
            self.cloned_stride = self._stride
            return self

    frame = FakeFrame(strided=True)
    source = object.__new__(MediaCaptureSource)
    source._nvdec_decoder = SimpleNamespace(get_batch_frames=lambda _count: [frame])
    source._nvdec_torch = SimpleNamespace(
        uint16=frame.dtype,
        from_dlpack=lambda value: value,
        cuda=SimpleNamespace(
            Stream=lambda **_kwargs: SimpleNamespace(
                __enter__=lambda self: self,
                __exit__=lambda self, *_args: None,
                synchronize=lambda: None,
            ),
            stream=lambda stream: nullcontext(stream),
            current_device=lambda: 0,
        ),
    )
    source._nvdec_buffer_size = 1
    source._nvdec_copy_stream = None

    result = source._decode_nvdec_frames(1)

    assert result is not frame
    assert result.cloned_stride == (6, 1)


def test_nvdec_batched_fetch_returns_owned_latest_gpu_frame():
    frames = []
    calls = []
    stream_calls = []
    stream_devices = []
    stream_state = {"active": False}

    class FakeFrame:
        def __init__(self, index):
            self.ndim = 3
            self.shape = (3, 2, 2)
            self.device = SimpleNamespace(type="cuda")
            self.index = index

        def clone(self):
            assert stream_state["active"]
            stream_calls.append("clone")
            return FakeFrame(self.index)

    class FakeStream:
        def __enter__(self):
            assert not stream_state["active"]
            stream_state["active"] = True
            stream_calls.append("enter")
            return self

        def __exit__(self, *_args):
            stream_state["active"] = False
            stream_calls.append("exit")

        def synchronize(self):
            assert not stream_state["active"]
            stream_calls.append("synchronize")

    def from_dlpack(frame):
        assert stream_state["active"]
        stream_calls.append("from_dlpack")
        return frame

    def create_stream(device):
        stream_devices.append(device)
        return FakeStream()

    class FakeDecoder:
        def get_batch_frames(self, count):
            calls.append(count)
            batch = []
            for _ in range(count):
                frame = FakeFrame(len(frames) + 1)
                frames.append(frame)
                batch.append(frame)
            return batch

    source = object.__new__(MediaCaptureSource)
    source._nvdec_decoder = FakeDecoder()
    source._nvdec_torch = SimpleNamespace(
        from_dlpack=from_dlpack,
        cuda=SimpleNamespace(
            Stream=create_stream,
            current_device=lambda: 0,
            stream=lambda copy_stream: copy_stream,
        ),
    )
    source._nvdec_buffer_size = 4

    frame = source._decode_nvdec_frames(6)
    next_frame = source._decode_nvdec_frames(1)

    assert calls == [4, 2, 1]
    assert frame is not frames[5]
    assert frame.index == 6
    assert next_frame.index == 7
    assert frame.device.type == "cuda"
    assert stream_devices == [0]
    assert stream_calls == [
        "enter", "from_dlpack", "clone", "exit", "synchronize",
        "enter", "from_dlpack", "clone", "exit", "synchronize",
    ]


def test_nvdec_downsampling_reads_kept_and_skipped_frames_in_one_batch():
    source = object.__new__(MediaCaptureSource)
    source._nvdec_first_frame_pending = False
    source._nvdec_next_frame_at = 0.0
    source._playback_fps = 60.0
    source._source_fps = 120.0
    source._nvdec_skip_fraction = 0.0
    source._nvdec_disabled = False
    source._frame = None
    calls = []
    source._decode_nvdec_frames = lambda count: calls.append(count) or "latest-frame"

    frame = source._read_nvdec_frame()

    assert calls == [2]
    assert frame == "latest-frame"
    assert source._nvdec_skip_fraction == 0.0


def test_close_nvdec_ends_threaded_decoder_and_stops_simple_decoder():
    calls = []
    source = object.__new__(MediaCaptureSource)
    source._nvdec_decoder = SimpleNamespace(
        end=lambda: calls.append("end"),
        stop=lambda: calls.append("stop"),
    )
    source._nvdec_torch = object()
    source._nvdec_first_frame_pending = True
    source._nvdec_next_frame_at = 1.0

    source._close_nvdec()

    assert calls == ["end"]
    assert source._nvdec_decoder is None
    assert source._nvdec_torch is None

    source._nvdec_decoder = SimpleNamespace(stop=lambda: calls.append("stop"))
    source._close_nvdec()

    assert calls == ["end", "stop"]


def test_video_decode_does_not_hide_corrupt_media_with_software_fallback():
    source = _source_for_decode_start()
    software_starts = []
    source._start_ffmpeg_video = lambda position, *, realtime: software_starts.append(position)
    source._start_nvdec = lambda position: (_ for _ in ()).throw(
        ValueError("corrupt packet data")
    )

    with pytest.raises(ValueError, match="corrupt packet"):
        source._start_video(0.0, realtime=True)

    assert software_starts == []
    assert source._nvdec_disabled is False


def test_rocm_torch_is_rejected_before_nvdec_decoder_import(monkeypatch):
    source = _source_for_decode_start()
    source._nvdec_source_path = "sample.mp4"
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip="7.0"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setattr("capture.media._enable_torch_cuda_dlls", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="ROCm is active"):
        source._start_nvdec(0.0)


def test_probe_media_retains_video_codec_name(monkeypatch):
    payload = {
        "streams": [{"codec_type": "video", "codec_name": "H264", "width": 2, "height": 1}],
        "format": {"duration": "3.0"},
    }
    monkeypatch.setattr(
        "capture.media.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    info = _probe_media("ffprobe", ["-i", "sample.mp4"])

    assert info["codec_name"] == "h264"


def test_4k_420_hdr_preserves_p010_for_gpu_preprocessing():
    info = {
        "width": 3840,
        "height": 2160,
        "pix_fmt": "yuv420p10le",
        "colorspace": "bt2020nc",
        "transfer": "smpte2084",
        "primaries": "bt2020",
        "color_range": "tv",
        "chroma_location": "left",
        "fps": 120.0,
    }

    pixel_format = _video_pixel_format(info)

    assert pixel_format == "p010le"
    assert _media_frame_format(info, pixel_format) == "p010le_pq_bt2020_bt2020nc_tv_left"
    assert _video_filter(info, 60) == "fps=60,format=p010le"


def test_hdr_media_uses_bt2020_defaults_when_container_tags_are_missing():
    info = {
        "pix_fmt": "yuv420p10le",
        "transfer": "smpte2084",
        "colorspace": "unknown",
        "primaries": "unknown",
    }

    assert (
        _media_frame_format(info, "p010le")
        == "p010le_pq_bt2020_bt2020nc_tv_left"
    )
    assert _media_frame_format(info, "bgr24") == "bgr24_pq_bt2020"


def test_8_bit_420_media_keeps_nv12_output():
    info = {
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "colorspace": "bt709",
        "transfer": "bt709",
        "primaries": "bt709",
        "color_range": "tv",
        "chroma_location": "left",
        "fps": 60.0,
    }

    assert _video_pixel_format(info) == "nv12"
    assert _video_filter(info, 60) == "fps=60,format=nv12"


def test_amf_vpp_downscales_hdr_without_changing_p010_samples_or_transfer():
    from stereo_runtime.render_size import RenderSizeConfig, RenderSizePolicy

    info = {
        "width": 3840,
        "height": 1920,
        "pix_fmt": "yuv420p10le",
        "colorspace": "bt2020nc",
        "transfer": "smpte2084",
        "primaries": "bt2020",
        "color_range": "tv",
        "chroma_location": "left",
        "fps": 120.0,
    }
    render_size = RenderSizeConfig(
        policy=RenderSizePolicy.FIXED, fixed_width=1920, fixed_height=960
    )

    output_size = _resolve_amf_output_size(info, render_size)
    video_filter = _amf_vpp_filter(info, 60, output_size)

    assert output_size is None
    assert video_filter == ""


@pytest.mark.parametrize(
    ("scale_factor", "expected_size"),
    [
        ("100%", None),
        ("85%", (3264, 1836)),
        ("75%", (2880, 1620)),
        ("50%", (1920, 1080)),
    ],
)
def test_amf_vpp_only_changes_source_dimensions_for_ui_quality_scale(
    scale_factor, expected_size
):
    from stereo_runtime.render_size import RenderSizeConfig, RenderSizePolicy

    info = {
        "width": 3840,
        "height": 2160,
        "pix_fmt": "yuv420p",
        "colorspace": "bt709",
        "transfer": "bt709",
        "primaries": "bt709",
        "color_range": "tv",
        "chroma_location": "left",
        "fps": 60.0,
    }
    render_size = RenderSizeConfig(
        policy=RenderSizePolicy.SCALED,
        scale_factor=scale_factor,
    )

    output_size = _resolve_amf_output_size(info, render_size)

    assert output_size == expected_size
    assert bool(_amf_vpp_filter(info, 60, output_size)) is (expected_size is not None)


def test_amf_vpp_downscales_sdr_to_nv12_and_skips_unsupported_targets():
    from stereo_runtime.render_size import RenderSizeConfig, RenderSizePolicy

    info = {
        "width": 3840,
        "height": 2160,
        "pix_fmt": "yuv420p",
        "colorspace": "bt709",
        "transfer": "bt709",
        "primaries": "bt709",
        "color_range": "tv",
        "chroma_location": "left",
        "fps": 60.0,
    }
    smaller = RenderSizeConfig(
        policy=RenderSizePolicy.FIXED, fixed_width=1920, fixed_height=1080
    )
    larger = RenderSizeConfig(
        policy=RenderSizePolicy.FIXED, fixed_width=4096, fixed_height=2304
    )

    output_size = _resolve_amf_output_size(info, smaller)

    assert output_size == (1920, 1080)
    assert _amf_vpp_filter(info, 60, output_size) == (
        "vpp_amf=w=1920:h=1080:format=nv12,hwdownload,format=nv12,fps=60"
    )
    assert _resolve_amf_output_size(info, larger) is None
    assert _resolve_amf_output_size(info, None) is None
    assert _amf_vpp_filter({**info, "pix_fmt": "yuv422p"}, 60, (1920, 1080)) == ""


def test_p010_media_uses_nvdec_native_on_nvidia():
    source = object.__new__(MediaCaptureSource)
    source._video_process = None
    source._nvdec_source_path = "sample.mp4"
    source._nvdec_disabled = False
    source._gpu_vendor = "nvidia"
    source._frame_pixel_format = "p010le"
    source._frame_format = "p010le_pq_bt2020_bt2020nc_tv_left"
    source._ffmpeg_frame_format = source._frame_format
    source._nvdec_decoder = None
    source._nvdec_torch = None
    source._amf_decoder = "hevc_amf"
    source._amf_disabled = False
    source.hdr = True
    source.transfer = "smpte2084"
    source.primaries = "bt2020"
    source._close_nvdec = lambda: None
    source._close_opencv = lambda: None
    starts = []
    native = []

    def start_nvdec(position, *, native_hdr=False):
        native.append((position, native_hdr))
        source._decode_backend = "nvdec-native-p010"

    def start_ffmpeg(position, *, realtime, decoder=None):
        starts.append((position, realtime, decoder))

    source._start_nvdec = start_nvdec
    source._start_ffmpeg_video = start_ffmpeg
    source._start_video(2.0, realtime=True)

    assert native == [(2.0, True)]
    assert starts == []
    assert source._decode_backend == "nvdec-native-p010"


def test_ffmpeg_nv12_output_is_read_as_two_plane_frame():
    source = object.__new__(MediaCaptureSource)
    source._decode_backend = "ffmpeg-software"
    source._frame_pixel_format = "nv12"
    source._video_process = SimpleNamespace(
        stdout=io.BytesIO(bytes([16, 32, 48, 64, 128, 128]))
    )
    source.width = 2
    source.height = 2

    frame = source._read_video_frame()

    assert frame.shape == (3, 2)
    assert frame.tolist() == [[16, 32], [48, 64], [128, 128]]


def test_ffmpeg_p010_output_is_read_as_two_plane_uint16_frame():
    source = object.__new__(MediaCaptureSource)
    source._decode_backend = "ffmpeg-software"
    source._frame_pixel_format = "p010le"
    values = np.array(
        [[64 << 6, 65 << 6], [66 << 6, 67 << 6], [512 << 6, 512 << 6]],
        dtype="<u2",
    )
    source._video_process = SimpleNamespace(stdout=io.BytesIO(values.tobytes()))
    source.width = 2
    source.height = 2

    frame = source._read_video_frame()

    assert frame.shape == (3, 2)
    assert frame.dtype == np.dtype("<u2")
    assert frame.tolist() == values.tolist()


def test_amf_scaled_p010_frame_uses_decoded_dimensions():
    source = object.__new__(MediaCaptureSource)
    source._decode_backend = "ffmpeg-software"
    source._frame_pixel_format = "p010le"
    source.width = 4
    source.height = 4
    source._decode_width = 2
    source._decode_height = 2
    values = np.arange(6, dtype="<u2")
    source._video_process = SimpleNamespace(stdout=io.BytesIO(values.tobytes()))

    frame = source._read_video_frame()

    assert frame.shape == (3, 2)
    assert frame.tolist() == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.parametrize(
    ("transfer", "primaries", "frame_format"),
    [
        ("smpte2084", "smpte432", "bgr24_pq_smpte432"),
        ("arib-std-b67", "bt2020", "bgr24_hlg_bt2020"),
    ],
)
def test_hdr_ffmpeg_path_preserves_transfer_for_device_preprocess(
    transfer, primaries, frame_format, monkeypatch
):
    monkeypatch.setattr("capture.media.sys.platform", "linux")
    source = _source_for_decode_start()
    source.hdr = True
    source.transfer = transfer
    source.primaries = primaries
    source._start_nvdec = lambda _position: (_ for _ in ()).throw(
        RuntimeError("hardware decoder unavailable")
    )
    starts = []
    source._start_ffmpeg_video = lambda position, *, realtime: starts.append((position, realtime))

    source._start_video(2.5, realtime=True)
    video_filter = _video_filter(
        {"transfer": transfer, "primaries": primaries, "fps": 23.976}, 30
    )

    assert source.frame_format == frame_format
    assert starts == [(2.5, True)]
    assert video_filter.startswith("fps=23.976,format=bgr24")
    assert "tonemap" not in video_filter
    assert "geq" not in video_filter


def test_amf_decoder_failure_restarts_with_ffmpeg_software_decode():
    source = _source_for_ffmpeg_failure("Unknown decoder 'h264_amf'")
    source._decode_backend = "ffmpeg-amf"
    source._amf_disabled = False
    source._has_audio = False
    source._current_position = lambda: 2.0
    restart_positions = []

    def start_software(position, *, realtime):
        restart_positions.append((position, realtime))
        source._decode_backend = "ffmpeg-software"
        source._video_process = SimpleNamespace(
            stdout=io.BytesIO(bytes([1, 2, 3])),
            poll=lambda: None,
        )

    source._start_video = start_software

    frame = source._read_video_frame()

    assert frame.tolist() == [[[1, 2, 3]]]
    assert restart_positions == [(2.0, True)]
    assert source._amf_disabled is True


def test_amf_process_crash_falls_back_to_software_decode():
    source = _source_for_ffmpeg_failure("")
    source._decode_backend = "ffmpeg-amf"
    source._amf_disabled = False
    source._has_audio = False
    source._current_position = lambda: 2.0
    restart_positions = []

    def start_software(position, *, realtime):
        restart_positions.append((position, realtime))
        source._decode_backend = "ffmpeg-software"
        source._video_process = SimpleNamespace(
            stdout=io.BytesIO(bytes([1, 2, 3])),
            poll=lambda: None,
        )

    source._start_video = start_software

    frame = source._read_video_frame()

    assert frame.tolist() == [[[1, 2, 3]]]
    assert restart_positions == [(2.0, True)]
    assert source._amf_disabled is True
    assert source.decode_fallback_reason == "FFmpeg AMF decoder exited with status 1"


def test_amf_vpp_format_negotiation_failure_falls_back_to_software():
    source = _source_for_ffmpeg_failure(
        "[Parsed_vpp_amf_0 @ 0x1] Impossible to convert between the formats "
        "supported by the filter and Parsed_hwdownload_1"
    )
    source._decode_backend = "ffmpeg-amf"
    source._amf_disabled = False
    source._has_audio = False
    source._current_position = lambda: 2.0
    restart_positions = []

    def start_software(position, *, realtime):
        restart_positions.append((position, realtime))
        source._decode_backend = "ffmpeg-software"
        source._video_process = SimpleNamespace(
            stdout=io.BytesIO(bytes([1, 2, 3])),
            poll=lambda: None,
        )

    source._start_video = start_software

    frame = source._read_video_frame()

    assert frame.tolist() == [[[1, 2, 3]]]
    assert restart_positions == [(2.0, True)]
    assert source._amf_disabled is True
    assert "impossible to convert" in source.decode_fallback_reason.casefold()


class _ExitedProcess:
    def __init__(self, stderr_text):
        self.stdout = io.BytesIO()
        self.returncode = 1
        self._media_stderr = tempfile.TemporaryFile()
        self._media_stderr.write(stderr_text.encode("utf-8"))
        self._media_stderr.flush()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 1

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = 1


def _source_for_ffmpeg_failure(stderr_text):
    source = object.__new__(MediaCaptureSource)
    source.width = 1
    source.height = 1
    source._decode_backend = "ffmpeg-software"
    source._video_process = _ExitedProcess(stderr_text)
    source._nvdec_source_path = "sample.mp4"
    source._current_position = lambda: 2.0
    source._position_anchor = 1.25
    source._frame = None
    source._playback_fps = 30.0
    source._source_fps = 30.0
    source.playing = True
    source.decode_fallback_reason = ""
    source.error = ""
    return source


@pytest.mark.parametrize(
    "stderr_text",
    [
        "Your platform doesn't support hardware accelerated AV1 decoding",
        "Decoder not found for codec av1",
    ],
)
def test_ffmpeg_decoder_unavailable_falls_back_to_opencv_cpu_decoder(stderr_text):
    source = _source_for_ffmpeg_failure(stderr_text)
    fallback_positions = []
    source._start_opencv_video = lambda position, *, realtime: fallback_positions.append(
        (position, realtime)
    )
    source._read_opencv_frame = lambda: "cpu-frame"

    assert source._read_video_frame() == "cpu-frame"
    assert fallback_positions == [(1.25, True)]
    assert source._video_process is None


def test_ffmpeg_corrupt_media_does_not_fall_back_to_opencv():
    source = _source_for_ffmpeg_failure("Invalid data found when processing input")
    fallback_positions = []
    source._start_opencv_video = lambda position, *, realtime: fallback_positions.append(
        (position, realtime)
    )

    assert source._read_video_frame() is None
    assert fallback_positions == []
    assert "Invalid data" in source.error


def test_media_controls_track_revision_per_channel_and_matching_session(tmp_path):
    openxr_control_path = tmp_path / "openxr-control.json"
    gui_control_path = tmp_path / "gui-control.json"
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(
        media_control_path=str(openxr_control_path),
        media_gui_control_path=str(gui_control_path),
    )
    source._session_id = "session-123"
    source._last_control_revisions = {}
    source._image = False
    source.duration = 30.0
    source.playing = False
    source._current_position = lambda: 4.0
    actions = []
    source._restart = lambda position, *, playing: actions.append((position, playing))
    source._publish_state = lambda **_kwargs: None

    def send(path, session, revision, action):
        path.write_text(
            json.dumps({"session": session, "revision": revision, "action": action}),
            encoding="utf-8",
        )
        source._read_control()

    send(gui_control_path, "other-session", 1, "play")
    assert actions == []
    send(openxr_control_path, "session-123", 1, "play")
    send(gui_control_path, "session-123", 1, "pause")
    send(gui_control_path, "session-123", 1, "play")

    assert actions == [(4.0, True), (4.0, False)]


def test_browse_select_keeps_media_and_browse_close_restores_last_frame(tmp_path):
    control_path = tmp_path / "viewer-control.json"
    selected = []
    closed = []
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(
        media_control_path=None,
        media_gui_control_path=None,
        media_viewer_control_path=str(control_path),
    )
    source._session_id = "session-123"
    source._last_control_revisions = {}
    source._browser = SimpleNamespace(
        select=lambda path: selected.append(path) or True,
        close=lambda: closed.append(True),
    )
    source._browser_mode = True
    source._image = False
    source._ready = True
    source.playing = False
    source._frame = object()
    source.width = 4
    source.height = 2
    source._publish_state = lambda **_kwargs: None

    def send(revision, action, value=None):
        request = {
            "session": "session-123",
            "revision": revision,
            "action": action,
        }
        if value is not None:
            request["value"] = value
        control_path.write_text(json.dumps(request), encoding="utf-8")
        source._read_control()

    send(1, "browse_select", {"path": "F:/Videos/b.mp4"})
    assert selected == ["F:/Videos/b.mp4"]
    assert source._browser_mode is True

    send(2, "browse_close")
    assert closed == [True]
    assert source._browser_mode is False
    frame, _size = source.grab()
    assert frame is source._frame


def test_media_exit_stops_and_opens_current_file_directory(tmp_path):
    control_path = tmp_path / "viewer-control.json"
    media_path = tmp_path / "Movies" / "movie.mp4"
    media_path.parent.mkdir()
    opened = []
    stopped = []
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(
        media_control_path=None,
        media_gui_control_path=None,
        media_viewer_control_path=str(control_path),
    )
    source._session_id = "session-456"
    source._last_control_revisions = {}
    source._browser = SimpleNamespace(open=lambda path: opened.append(path))
    source.path = str(media_path)
    source._browser_mode = False
    source._frame = object()
    source._displayed_frame = object()
    source.playing = True
    source._ready = True
    source.stop = lambda: stopped.append(True)
    source._publish_state = lambda **_kwargs: None
    control_path.write_text(
        json.dumps(
            {
                "session": "session-456",
                "revision": 1,
                "action": "media_exit",
            }
        ),
        encoding="utf-8",
    )

    source._read_control()

    assert stopped == [True]
    assert opened == [str(media_path.parent)]
    assert source._browser_mode is True
    assert source._frame is None
    assert source._displayed_frame is None
    assert source.playing is False


def test_playlist_auto_advance_returns_next_frame_not_grab_tuple():
    next_frame = object()
    source = object.__new__(MediaCaptureSource)
    source._browser = SimpleNamespace(playlist_step=lambda _direction: "next.mp4")
    source._open_playlist_path = lambda path: path == "next.mp4"
    source._stop_frame_pump = lambda: None
    source._close_native_decoder = lambda: None
    source._close_nvdec = lambda: None
    source._close_opencv = lambda: None
    source._close_audio = lambda: None
    source._video_process = None
    source._terminate = lambda _process: None
    source.grab = lambda: (next_frame, (4, 2))

    assert source._finish_video_stream() is next_frame


def test_video_eof_returns_last_decoded_frame_to_presenter():
    last_frame = object()
    source = object.__new__(MediaCaptureSource)
    source._ready = True
    source._browser_mode = False
    source._image = False
    source.playing = True
    source._frame = last_frame
    source._eof = False
    source.duration = 12.0
    source.position = 0.0
    source.width = 4
    source.height = 2
    source._decode_backend = "nvdec"
    source._video_process = None
    source._browser = SimpleNamespace(playlist_step=lambda _direction: None)
    source._read_control = lambda: None
    source._read_video_frame = lambda: None
    source._terminate = lambda _process: None
    source._close_nvdec = lambda: None
    source._close_opencv = lambda: None
    source._close_audio = lambda: None
    source._current_position = lambda: 12.0
    source._publish_state = lambda **_kwargs: None

    frame, _size = source.grab()

    assert frame is last_frame
    assert source.playing is False
    assert source._eof is True


def test_open_media_command_dispatches_path_mode_and_resume_position(tmp_path):
    control_path = tmp_path / "gui-control.json"
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(
        media_control_path=None,
        media_gui_control_path=str(control_path),
    )
    source._session_id = "session-123"
    source._last_control_revisions = {}
    source._image = False
    source.duration = 30.0
    source.playing = False
    source._current_position = lambda: 4.0
    opened = []
    source._open_control_media = lambda value: opened.append(value) or True
    source._publish_state = lambda **_kwargs: None
    media = {"path": "D:/media/next.mkv", "mode": "Video", "position": 12.5}
    control_path.write_text(
        json.dumps({
            "session": "session-123",
            "revision": 1,
            "action": "open",
            "value": media,
        }),
        encoding="utf-8",
    )

    source._read_control()

    assert opened == [media]


def test_open_control_media_reinitializes_video_and_image():
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(fps=30)
    source.path = "D:/media/old.mp4"
    source.mode = "video"
    source._image = False
    source.position = 3.0
    source._current_position = lambda: source.position
    stopped = []
    source.stop = lambda: stopped.append(True)
    opened = []

    def open_source():
        opened.append((source.path, source.mode, source.position, source._has_audio))
        source._ready = True
        source.playing = source.mode == "video"
        if source.mode == "image":
            source.position = 0.0

    source._open = open_source

    assert source._open_control_media({
        "path": "D:/media/movie.mkv", "mode": "Video", "position": 12.5,
    })
    assert opened[-1] == ("D:/media/movie.mkv", "video", 12.5, False)
    assert source.playing is True
    assert source._ready is True

    assert source._open_control_media({
        "path": "D:/media/still.png", "mode": "Image", "position": 12.5,
    })
    assert opened[-1] == ("D:/media/still.png", "image", 0.0, False)
    assert source.playing is False
    assert source._ready is True
    assert len(stopped) == 2


def test_open_control_media_reports_unavailable_path_without_raising():
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(fps=30)
    source.path = "D:/media/old.mp4"
    source.mode = "video"
    source._image = False
    source.position = 3.0
    source._current_position = lambda: source.position
    source.stop = lambda: None
    source._open = lambda: (_ for _ in ()).throw(
        FileNotFoundError("Media source does not exist")
    )

    assert source._open_control_media({
        "path": "D:/missing.mp4", "mode": "Video", "position": 0.0,
    })
    assert source.error.startswith("FileNotFoundError:")
    assert source.playing is False
    assert source._ready is False


def test_open_media_command_ignores_other_session(tmp_path):
    control_path = tmp_path / "gui-control.json"
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(
        media_control_path=None,
        media_gui_control_path=str(control_path),
    )
    source._session_id = "session-123"
    source._last_control_revisions = {}
    source._image = False
    source.duration = 30.0
    source.playing = False
    source._current_position = lambda: 4.0
    opened = []
    source._open_control_media = lambda value: opened.append(value) or True
    source._publish_state = lambda **_kwargs: None
    control_path.write_text(
        json.dumps({
            "session": "other-session",
            "revision": 1,
            "action": "open",
            "value": {"path": "D:/media/next.mkv", "mode": "Video"},
        }),
        encoding="utf-8",
    )

    source._read_control()

    assert opened == []


def test_openxr_media_command_writer_serializes_session_revision_and_value(tmp_path):
    from app_runtime.runtime_callbacks import RuntimeCallbacks

    control_path = tmp_path / "openxr-control.json"
    writer = object.__new__(RuntimeCallbacks)
    writer.context = SimpleNamespace(capture_config=SimpleNamespace(
        media_control_path=str(control_path),
        media_session_id="session-123",
    ))
    writer._media_command_revision = -1

    assert writer._write_openxr_media_command("seek", 14.25) is True
    command = json.loads(control_path.read_text(encoding="utf-8"))
    assert command == {
        "session": "session-123",
        "revision": 0,
        "action": "seek",
        "value": 14.25,
    }

    assert writer._write_openxr_media_command("seek", float("nan")) is False
    assert writer._media_command_revision == 0
    assert json.loads(control_path.read_text(encoding="utf-8")) == command

    assert writer._write_openxr_media_command(
        "browse_select", {"path": "F:/Videos/movie.mp4"}
    ) is True
    command = json.loads(control_path.read_text(encoding="utf-8"))
    assert command == {
        "session": "session-123",
        "revision": 1,
        "action": "browse_select",
        "value": {"path": "F:/Videos/movie.mp4"},
    }

    assert writer._write_openxr_media_command(
        "browser_sort", {"descending": True}
    ) is True
    command = json.loads(control_path.read_text(encoding="utf-8"))
    assert command == {
        "session": "session-123",
        "revision": 2,
        "action": "browser_sort",
        "value": {"descending": True},
    }


def test_preferred_audio_output_device_prefers_windows_wasapi(monkeypatch):
    class SoundDevice:
        @staticmethod
        def query_hostapis():
            return [
                {"name": "MME", "default_output_device": 13},
                {"name": "Windows WASAPI", "default_output_device": 40},
            ]

    monkeypatch.setattr("capture.media.os.name", "nt")
    monkeypatch.delenv("D2S_MEDIA_AUDIO_DEVICE", raising=False)

    assert _preferred_audio_output_device(SoundDevice) == 40

    monkeypatch.setenv("D2S_MEDIA_AUDIO_DEVICE", "17")
    assert _preferred_audio_output_device(SoundDevice) == 17


def test_audio_loop_records_device_failure_without_thread_exception():
    class Stream:
        def write(self, _samples):
            raise RuntimeError("no audio driver")

        def stop(self):
            pass

        def close(self):
            pass

    process = SimpleNamespace(
        stdout=io.BytesIO(np.zeros(4096, dtype=np.float32).tobytes()),
        terminated=False,
        poll=lambda: None,
        terminate=lambda: setattr(process, "terminated", True),
    )
    source = object.__new__(MediaCaptureSource)
    source._audio_stop = threading.Event()
    source._audio_process = process
    source._audio_stream = Stream()
    source.volume = 1.0
    source.muted = False
    source.audio_error = ""
    source._audio_loop()

    assert "no audio driver" in source.audio_error
    assert source._audio_stop.is_set()
    assert source._audio_process is None
    assert source._audio_stream is None
    assert process.terminated is True


def test_media_state_publishes_session_and_playback_facts(tmp_path):
    state_path = tmp_path / "state.json"
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_state_path=str(state_path))
    source._session_id = "session-123"
    source._last_publish_at = 0.0
    source.path = "D:/media/movie.mkv"
    source._image = False
    source._ready = True
    source.position = 4.0
    source.duration = 30.0
    source.playing = True
    source.volume = 0.75
    source.muted = False
    source.hdr = True
    source._decode_backend = "ffmpeg-software"
    source.decode_fallback_reason = ""
    source._has_audio = True
    source.error = ""
    source.audio_error = ""
    source._current_position = lambda: 5.5

    source._publish_state(force=True)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["session"] == "session-123"
    assert state["position"] == 5.5
    assert state["duration"] == 30.0
    assert state["playing"] is True
    assert state["ready"] is True
    assert state["volume"] == 0.75
    assert state["hdr"] is True
    assert state["decode_backend"] == "ffmpeg-software"


def test_media_state_retries_transient_replace_permission_error(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_state_path=str(state_path))
    source._session_id = "session-123"
    source._last_publish_at = 0.0
    source.path = "D:/media/movie.mkv"
    source._image = False
    source.position = 4.0
    source.duration = 30.0
    source.playing = False
    source.volume = 1.0
    source.muted = False
    source.hdr = False
    source._decode_backend = "ffmpeg-software"
    source.decode_fallback_reason = ""
    source._has_audio = False
    source.error = "Could not save media state: previous transient failure"
    source.audio_error = ""
    source._current_position = lambda: 4.0

    replace = os.replace
    attempts = []

    def flaky_replace(source_path, destination_path):
        attempts.append(True)
        if len(attempts) < 3:
            raise PermissionError(13, "access denied")
        replace(source_path, destination_path)

    monkeypatch.setattr("capture.media.os.replace", flaky_replace)

    source._publish_state(force=True)

    assert len(attempts) == 3
    assert source.error == ""
    assert json.loads(state_path.read_text(encoding="utf-8"))["position"] == 4.0



def test_open_control_media_restores_previous_media_when_new_media_fails():
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(fps=30)
    source.path = "D:/media/current.mp4"
    source.mode = "video"
    source._image = False
    source.position = 4.0
    source.playing = True
    source._current_position = lambda: 9.25
    stop_calls = []
    source.stop = lambda: stop_calls.append(True)
    opened_paths = []

    def open_source():
        opened_paths.append(source.path)
        if source.path.endswith("missing.mp4"):
            raise FileNotFoundError("Media source does not exist")
        source._ready = True
        source.playing = True

    source._open = open_source

    assert source._open_control_media({
        "path": "D:/media/missing.mp4", "mode": "Video", "position": 0.0,
    })

    assert opened_paths == ["D:/media/missing.mp4", "D:/media/current.mp4"]
    assert len(stop_calls) == 2
    assert source.path == "D:/media/current.mp4"
    assert source.position == 9.25
    assert source.playing is True
    assert source._ready is True
    assert source.error.startswith("FileNotFoundError:")



def test_nvdec_runtime_fallback_restarts_audio_at_video_position():
    source = object.__new__(MediaCaptureSource)
    source._nvdec_first_frame_pending = False
    source._nvdec_next_frame_at = None
    source._playback_fps = 30.0
    source._source_fps = 30.0
    source._nvdec_skip_fraction = 0.0
    source._nvdec_disabled = False
    source._frame = None
    source._has_audio = True
    source.playing = True
    source._decode_nvdec_frame = lambda: (_ for _ in ()).throw(
        RuntimeError("hardware decoder unavailable")
    )
    source._current_position = lambda: 12.5
    calls = []
    source._close_audio = lambda: calls.append(("close-audio",))
    source._start_video = lambda position, *, realtime: calls.append(
        ("start-video", position, realtime)
    )
    source._start_audio = lambda position: calls.append(("start-audio", position))
    source._read_video_frame = lambda: "software-frame"

    assert source._read_nvdec_frame() == "software-frame"
    assert calls == [
        ("close-audio",),
        ("start-video", 12.5, True),
        ("start-audio", 12.5),
    ]


@pytest.mark.parametrize("position", [float("nan"), float("inf"), float("-inf")])
def test_seek_ignores_non_finite_positions(tmp_path, position):
    control_path = tmp_path / "media-control.json"
    control_path.write_text(json.dumps({
        "revision": 1,
        "action": "seek",
        "value": position,
    }), encoding="utf-8")
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(
        media_control_path=str(control_path), media_gui_control_path=None
    )
    source._session_id = ""
    source._last_control_revisions = {}
    source._image = False
    restarts = []
    source._restart = lambda value, *, playing: restarts.append((value, playing))
    source._publish_state = lambda **_kwargs: None

    source._read_control()

    assert restarts == []


def test_unknown_duration_eof_persists_live_playback_position():
    source = object.__new__(MediaCaptureSource)
    source._ready = True
    source._image = False
    source.playing = True
    source.duration = 0.0
    source.position = 3.0
    source.width = 640
    source.height = 360
    source._video_process = None
    source._read_control = lambda: None
    source._read_video_frame = lambda: None
    source._close_nvdec = lambda: None
    source._close_opencv = lambda: None
    source._close_audio = lambda: None
    source._current_position = lambda: 8.0 if source.playing else source.position
    source._publish_state = lambda **_kwargs: None

    frame, size = source.grab()

    assert frame is None
    assert size == (640, 360)
    assert source.playing is False
    assert source.position == 8.0
    assert source._position_anchor == 8.0


def test_unknown_duration_play_after_eof_restarts_from_zero(tmp_path):
    control_path = tmp_path / "media-control.json"
    control_path.write_text(
        json.dumps({"revision": 1, "action": "play"}),
        encoding="utf-8",
    )
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(
        media_control_path=str(control_path), media_gui_control_path=None
    )
    source._session_id = ""
    source._last_control_revisions = {}
    source._image = False
    source._eof = True
    source.duration = 0.0
    source.position = 8.0
    source._current_position = lambda: source.position
    source._publish_state = lambda **_kwargs: None
    restarts = []
    source._restart = lambda value, *, playing: restarts.append((value, playing))

    source._read_control()

    assert restarts == [(0.0, True)]


def test_media_playback_clock_uses_high_resolution_perf_counter(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source.playing = True
    source.position = 0.0
    source.duration = 0.0
    source._position_anchor = 10.0
    source._position_anchor_time = 100.0

    monkeypatch.setattr("capture.media.time.perf_counter", lambda: 100.25)
    monkeypatch.setattr(
        "capture.media.time.monotonic",
        lambda: (_ for _ in ()).throw(AssertionError("media clock must use QPC")),
    )

    assert source._current_position() == pytest.approx(10.25)


def test_120fps_buffer_yields_deterministic_60hz_source_frames():
    source = object.__new__(MediaCaptureSource)
    source._playback_fps = 60.0
    source._source_fps = 120.0
    source._last_source_frame_duration = 1.0 / 120.0
    source._frame_buffer = deque(
        (index / 120.0, f"frame-{index}", index)
        for index in range(20)
    )
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_done = False

    first = source._take_buffered_frame(0.0, allow_first=True)
    selected = [first[2]]
    for tick in range(1, 8):
        item = source._take_buffered_frame(tick / 60.0)
        assert item is not None
        selected.append(item[2])

    assert selected == [0, 2, 4, 6, 8, 10, 12, 14]


@pytest.mark.parametrize("playback_fps", [59, 60, 72, 90, 120])
def test_120fps_selection_tracks_source_pts_at_output_tick(playback_fps):
    source = object.__new__(MediaCaptureSource)
    source._playback_fps = float(playback_fps)
    source._source_fps = 120.0
    source._last_source_frame_duration = 1.0 / 120.0
    source._frame_buffer = deque(
        (index / 120.0, f"frame-{index}", index)
        for index in range(400)
    )
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_done = False

    first = source._take_buffered_frame(0.0, allow_first=True)
    assert first[2] == 0
    for tick in range(1, 80):
        item = source._take_buffered_frame(tick / playback_fps)
        assert item is not None
        assert item[2] == (tick * 120) // playback_fps


def test_120fps_buffer_waits_for_exact_sample_instead_of_returning_stale_frame():
    source = object.__new__(MediaCaptureSource)
    source._playback_fps = 60.0
    source._source_fps = 120.0
    source._last_source_frame_duration = 1.0 / 120.0
    source._frame_buffer_condition = threading.Condition()
    source._frame_buffer = deque([(1.0 / 120.0, "frame-1", 1)])
    source._frame_pump_done = False

    def append_exact_frame() -> None:
        time.sleep(0.002)
        with source._frame_buffer_condition:
            source._frame_buffer.append((2.0 / 120.0, "frame-2", 2))
            source._frame_buffer_condition.notify_all()

    producer = threading.Thread(target=append_exact_frame)
    producer.start()
    try:
        item = source._take_buffered_frame(2.0 / 120.0)
    finally:
        producer.join(timeout=1.0)

    assert item is not None
    assert item[2] == 2


def test_buffered_playback_preserves_small_source_order_slip() -> None:
    source = object.__new__(MediaCaptureSource)
    source._playback_fps = 60.0
    source._source_fps = 60.0
    source._last_source_frame_duration = 1.0 / 60.0
    source._frame_buffer_stale_drops = 0
    source._frame_buffer_drop_log_at = 0.0
    source._frame_buffer = deque(
        [
            (0.0, "frame-0", 0),
            (1.0 / 60.0, "frame-1", 1),
            (2.0 / 60.0, "frame-2", 2),
        ]
    )
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_done = False

    item = source._take_buffered_frame(2.0 / 60.0)

    assert item[1] == "frame-0"
    assert item[2] == 0
    assert source._frame_buffer_stale_drops == 0
    assert list(source._frame_buffer) == [
        (1.0 / 60.0, "frame-1", 1),
        (2.0 / 60.0, "frame-2", 2),
    ]


def test_buffered_playback_bounds_large_exact_cadence_lag() -> None:
    source = object.__new__(MediaCaptureSource)
    source._playback_fps = 60.0
    source._source_fps = 60.0
    source._last_source_frame_duration = 1.0 / 60.0
    source._frame_buffer_stale_drops = 0
    source._frame_buffer_drop_log_at = 0.0
    source._frame_buffer = deque(
        (index / 60.0, f"frame-{index}", index) for index in range(8)
    )
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_done = False

    item = source._take_buffered_frame(7.0 / 60.0)

    assert item[1] == "frame-4"
    assert item[2] == 4
    assert source._frame_buffer_stale_drops == 4
    assert [entry[2] for entry in source._frame_buffer] == [5, 6, 7]


def test_media_initial_cadence_uses_requested_ceiling_not_fixed_40fps(monkeypatch):
    monkeypatch.delenv("D2S_MEDIA_DECIMATION_TARGET_FPS", raising=False)
    reset_media_playback_cadence_target()

    assert _resolve_media_playback_cadence(120.0, 3840, 1920, 120.0) == (120.0, 1)
    assert _resolve_media_playback_cadence(120.0, 3840, 1920, 60.0) == (60.0, 2)
    assert _resolve_media_playback_cadence(60.0, 1920, 1080, 60.0, target_fps=29.0) == (20.0, 3)
    monkeypatch.setenv("D2S_MEDIA_DECIMATION_TARGET_FPS", "40")
    assert _resolve_media_playback_cadence(120.0, 3840, 1920, 120.0) == (40.0, 3)


def test_media_cadence_matches_integral_display_ticks() -> None:
    select = _select_media_cadence_fps_for_display

    assert select(120.0, 60.0, 40.0) == 30.0
    assert select(120.0, 72.0, 40.0) == 36.0
    assert select(120.0, 80.0, 40.0) == 40.0
    assert select(120.0, 90.0, 40.0) == 30.0
    assert select(120.0, 120.0, 40.0) == 40.0


def test_frame_pump_applies_runtime_cadence_target(monkeypatch) -> None:
    # Runtime cadence retargeting is an explicit opt-in. Keep this test on the
    # legacy controller path so it still exercises the target application.
    monkeypatch.setenv("D2S_MEDIA_CADENCE_ADAPTIVE", "1")
    source = object.__new__(MediaCaptureSource)
    source._source_fps = 120.0
    source.width = 3840
    source.height = 1920
    source.config = SimpleNamespace(fps=120.0)
    source._playback_fps = 40.0
    source._frame_pump_decimation_step = 3
    source._last_source_frame_duration = 1.0 / 40.0
    source._last_decimated_emit_at = 1.0
    source._frame_buffer = deque([(0.0, "stale", 0)])
    source._frame_buffer_condition = threading.Condition()

    try:
        set_media_playback_cadence_target(24.0)
        source._apply_media_playback_cadence_target()

        assert get_media_playback_cadence_target() == 24.0
        assert source._playback_fps == 24.0
        assert source._frame_pump_decimation_step == 5
        assert source._last_source_frame_duration == pytest.approx(1.0 / 24.0)
        assert source._last_decimated_emit_at == 0.0
        assert list(source._frame_buffer) == []
    finally:
        reset_media_playback_cadence_target()


def test_frame_pump_emits_every_third_4k120_source_frame(monkeypatch):
    monkeypatch.setenv("D2S_MEDIA_DECIMATION_TARGET_FPS", "40")
    source = object.__new__(MediaCaptureSource)
    source._image = False
    source._ready = True
    source.playing = True
    source._playback_fps, source._frame_pump_decimation_step = (
        _resolve_media_playback_cadence(120.0, 3840, 1920, 120.0)
    )
    source._frame_buffer_capacity = 8
    source._frame_buffer = deque()
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_stop = threading.Event()
    source._frame_pump_done = False
    source._frame_pump_error = ""
    source._frame_pump_position = 0.0
    source._source_frame_index = 0
    source._frame_pump_thread = None
    source._decode_lock = threading.RLock()
    source._frame_pump_read_ms = deque(maxlen=120)
    source._frame_pump_frame_times = deque(maxlen=120)
    source._frame_pump_metrics_lock = threading.Lock()
    next_source_index = [0]

    def read_frame():
        index = next_source_index[0]
        next_source_index[0] += 1
        source._native_frame_pts = index / 120.0
        source._native_frame_duration = 1.0 / 120.0
        return f"frame-{index}"

    source._read_video_frame = read_frame
    source._start_frame_pump()
    try:
        deadline = time.monotonic() + 1.0
        while len(source._frame_buffer) < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [item[2] for item in list(source._frame_buffer)[:6]] == [
            0,
            3,
            6,
            9,
            12,
            15,
        ]
    finally:
        source._stop_frame_pump()


def test_frame_pump_is_bounded_and_pause_keeps_buffered_frames():
    source = object.__new__(MediaCaptureSource)
    source._image = False
    source._ready = True
    source.playing = True
    source._playback_fps = 60.0
    source._frame_buffer_capacity = 2
    source._frame_buffer = deque()
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_stop = threading.Event()
    source._frame_pump_done = False
    source._frame_pump_error = ""
    source._frame_pump_position = 12.0
    source._source_frame_index = 0
    source._frame_pump_thread = None
    source._decode_lock = threading.RLock()
    source._read_video_frame = lambda: object()
    source._position_anchor = 12.0
    source._position_anchor_time = None
    source._current_position = lambda: 12.0
    source._close_audio = lambda: None

    source._start_frame_pump()
    try:
        deadline = time.monotonic() + 1.0
        while len(source._frame_buffer) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(source._frame_buffer) == 2
    finally:
        source._pause_playback()

    assert source.playing is False
    assert len(source._frame_buffer) == 2
    assert all(isinstance(item[0], float) for item in source._frame_buffer)


def test_wall_clock_frame_pump_drops_oldest_when_consumer_is_slow():
    source = object.__new__(MediaCaptureSource)
    source._image = False
    source._ready = True
    source.playing = True
    source.config = SimpleNamespace(media_presentation_sync=True)
    source._source_fps = 60.0
    source._playback_fps = 60.0
    source._frame_pump_decimation_step = 1
    source._frame_buffer_capacity = 2
    source._frame_buffer = deque()
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_stop = threading.Event()
    source._frame_pump_done = False
    source._frame_pump_error = ""
    source._frame_pump_position = 12.0
    source._source_frame_index = 0
    source._frame_pump_thread = None
    source._decode_lock = threading.RLock()
    source._read_video_frame = lambda: object()
    source._position_anchor = 12.0
    source._position_anchor_time = None
    source._current_position = lambda: 13.0
    source._close_audio = lambda: None
    source._frame_buffer_stale_drops = 0
    source._frame_buffer_drop_log_at = 0.0

    source._start_frame_pump()
    try:
        deadline = time.monotonic() + 1.0
        while source._frame_buffer_stale_drops == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        source._stop_frame_pump()

    assert source._frame_buffer_stale_drops > 0
    assert len(source._frame_buffer) <= source._frame_buffer_capacity


def test_file_mode_without_saved_path_opens_output_browser_without_error(tmp_path):
    source = MediaCaptureSource(SimpleNamespace(
        media_path="",
        capture_mode="File",
        media_resume_position=0.0,
        media_volume=1.0,
        media_muted=False,
        media_browser_directory=str(tmp_path),
        media_session_id="browser-session",
        media_state_path=str(tmp_path / "media-state.json"),
        media_control_path=None,
        media_gui_control_path=None,
        media_viewer_control_path=None,
        media_audio_enabled=False,
        output_resolution=1080,
        fps=30,
    ))
    try:
        assert source._browser_mode is True
        assert source._ready is True
        assert source.error == ""
        assert source._browser.snapshot()["browser_visible"] is True
    finally:
        source.stop()


def test_file_mode_with_missing_saved_path_opens_configured_browser_directory(tmp_path):
    missing_path = tmp_path / "old-machine" / "movie.mp4"
    browser_directory = tmp_path / "media"
    browser_directory.mkdir()
    source = MediaCaptureSource(SimpleNamespace(
        media_path=str(missing_path),
        capture_mode="File",
        media_resume_position=0.0,
        media_volume=1.0,
        media_muted=False,
        media_browser_directory=str(browser_directory),
        media_session_id="browser-session",
        media_state_path=str(tmp_path / "media-state.json"),
        media_control_path=None,
        media_gui_control_path=None,
        media_viewer_control_path=None,
        media_audio_enabled=False,
        output_resolution=1080,
        fps=30,
    ))
    try:
        assert source._browser_mode is True
        assert source._ready is True
        assert source.error.startswith("FileNotFoundError:")
        assert source._browser.snapshot()["browser_path"] == str(browser_directory)
        assert source._browser.snapshot()["browser_visible"] is True
        frame, _size = source.grab()
        assert frame is None
    finally:
        source.stop()

@pytest.mark.parametrize("backend", ["nvdec", "ffmpeg-software", "source-frames"])
def test_frame_pump_decimates_once_and_preserves_original_pts(backend, monkeypatch):
    """Exercise the real reader + pump, not a mock that already invents correct PTS."""
    reset_media_playback_cadence_target()
    source = object.__new__(MediaCaptureSource)
    source._image = False
    source._ready = True
    source.playing = True
    source._source_fps = 120.0
    source._playback_fps = 40.0
    source._frame_pump_decimation_step = 3
    source._frame_buffer_capacity = 6
    source._frame_buffer = deque()
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_stop = threading.Event()
    source._frame_pump_done = False
    source._frame_pump_error = ""
    source._frame_pump_position = 15.0
    source._source_frame_index = 0
    source._frame_pump_thread = None
    source._decode_lock = threading.RLock()
    source._decode_backend = backend
    source._native_decoder = None
    source.width = source.height = 2
    source._frame = "frame-0"
    source._nvdec_first_frame_pending = True
    source._nvdec_next_frame_at = 0.0
    source._nvdec_skip_fraction = 0.0
    next_index = [1]

    def decode_batch(count):
        next_index[0] += count
        return f"frame-{next_index[0] - 1}"

    source._decode_nvdec_frames = decode_batch
    source._decode_nvdec_frame = lambda: decode_batch(1)
    if backend == "ffmpeg-software":
        source._ffmpeg_decimation_step = 3
        source._ffmpeg_first_frame_pending = True
        source._frame_pixel_format = "bgr24"
        source._video_process = SimpleNamespace(stdout=SimpleNamespace(
            readinto=lambda view: (view.__setitem__(slice(None), bytes(len(view))), len(view))[1]
        ))
    elif backend == "source-frames":
        source._read_video_frame = lambda: object()
    source._start_frame_pump()
    try:
        deadline = time.monotonic() + 1.0
        with source._frame_buffer_condition:
            while len(source._frame_buffer) < 6 and time.monotonic() < deadline:
                source._frame_buffer_condition.wait(0.01)
            frames = list(source._frame_buffer)
        assert source._frame_pump_error == ""
        assert [item[2] for item in frames] == [0, 3, 6, 9, 12, 15]
        assert [item[0] for item in frames] == pytest.approx([15.0 + index / 120.0 for index in range(0, 18, 3)])
        assert source._last_source_frame_duration == pytest.approx(1.0 / 40.0)
        assert all(item[3]["media_playback_fps"] == 40.0 for item in frames)
        source._playback_fps = 30.0
        assert all(item[3]["media_playback_fps"] == 40.0 for item in frames)
        if backend == "nvdec":
            assert [item[1] for item in frames] == [f"frame-{index}" for index in range(0, 18, 3)]
    finally:
        source._stop_frame_pump()


def test_ffmpeg_samples_before_transfer_without_second_fps_filter(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source._video_filter = "fps=60,format=p010le"
    source._frame_pump_decimation_step = 3
    source.width, source.height = 3840, 1920
    source._ffmpeg = "ffmpeg"
    source._input_args = ["-i", "media.mp4"]
    source._frame_pixel_format = "p010le"
    commands = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: commands.append(command) or SimpleNamespace())
    source._start_ffmpeg_video(0.0, realtime=True)
    try:
        video_filter = commands[0][commands[0].index("-vf") + 1]
        assert video_filter == "select='not(mod(n\\,3))',format=p010le"
        assert source._ffmpeg_decimation_step == 3
        assert source._ffmpeg_first_frame_pending
    finally:
        source._video_process._media_stderr.close()


def test_ffmpeg_120_to_36_uses_ordered_fractional_selector(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source._video_filter = "fps=60,format=p010le"
    source._source_fps = 120.0
    source._playback_fps = 36.0
    source._frame_pump_decimation_step = 3
    source.width, source.height = 3840, 1920
    source._ffmpeg = "ffmpeg"
    source._input_args = ["-i", "media.mp4"]
    source._frame_pixel_format = "p010le"
    commands = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: commands.append(command) or SimpleNamespace())
    source._start_ffmpeg_video(0.0, realtime=True)
    try:
        video_filter = commands[0][commands[0].index("-vf") + 1]
        assert video_filter == (
            "select='eq(mod(n\\,10)\\,0)+eq(mod(n\\,10)\\,3)+eq(mod(n\\,10)\\,6)',"
            "format=p010le"
        )
        assert source._ffmpeg_selection_active
    finally:
        source._video_process._media_stderr.close()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.0])
def test_display_cadence_rejects_invalid_rates(invalid):
    assert _select_media_cadence_fps_for_display(120.0, invalid, 40.0) is None


def test_decimated_media_pacing_keeps_absolute_phase_after_scheduler_jitter(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source._playback_fps = 30.0
    source._frame_pump_decimation_step = 4
    source._last_decimated_emit_at = 10.0
    source._last_source_frame_duration = 1.0 / 30.0
    source._frame_buffer = deque([(0.0, "first", 0), (1.0 / 30.0, "second", 4)])
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_done = False
    now = [10.0 + 1.0 / 30.0 + 0.002]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])
    assert source._take_buffered_frame(1.0) == (0.0, "first", 0)
    assert source._last_decimated_emit_at == pytest.approx(10.0 + 1.0 / 30.0)
    now[0] = 10.0 + 2.0 / 30.0
    assert source._take_buffered_frame(1.0) == (1.0 / 30.0, "second", 4)


def test_nvdec_buffered_reader_does_not_apply_second_wall_clock_pacer(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source._nvdec_first_frame_pending = False
    source._nvdec_next_frame_at = 100.0
    source._nvdec_skip_fraction = 0.0
    source._source_fps = 120.0
    source._playback_fps = 30.0
    source._frame_pump_thread = threading.current_thread()
    source._decode_nvdec_frames = lambda count: count
    monkeypatch.setattr("capture.media.time", SimpleNamespace(
        perf_counter=lambda: 99.0,
        sleep=lambda *_args: pytest.fail("pump readers must decode ahead"),
    ))
    assert source._read_nvdec_frame() == 4
    assert source._decoder_frame_advance == 4


def test_d3d11_media_decode_targets_source_slot_instead_of_wall_clock():
    source = object.__new__(MediaCaptureSource)
    calls = []
    source._native_decoder = SimpleNamespace(next_frame=lambda position: calls.append(position) or SimpleNamespace(
        pts_seconds=15.0 + 5.0 / 120.0, duration_seconds=1.0 / 120.0,
    ))
    source._native_frame_importer = SimpleNamespace(import_frame=lambda frame: "gpu-image")
    source._source_fps = 120.0
    source._frame_pump_decimation_step = 5
    source._source_frame_index = 1
    source._frame_pump_position = 15.0 + 1.0 / 120.0
    source._frame_pump_thread = threading.current_thread()
    source._current_position = lambda: 100.0
    assert source._read_d3d11_frame() == "gpu-image"
    assert calls == pytest.approx([15.0 + 5.5 / 120.0])
    assert source._decoder_frame_advance == 5
    assert source._decoder_resampled


def test_d3d11_media_uses_fractional_three_three_four_source_schedule():
    source = object.__new__(MediaCaptureSource)
    calls = []
    source._native_decoder = SimpleNamespace(
        next_frame=lambda position: calls.append(position) or SimpleNamespace(
            pts_seconds=15.0 + 5.0 / 120.0,
            duration_seconds=1.0 / 120.0,
        )
    )
    source._native_frame_importer = SimpleNamespace(import_frame=lambda frame: "gpu-image")
    source._source_fps = 120.0
    source._playback_fps = 36.0
    source._source_selection_schedule = None
    source._frame_pump_decimation_step = 3
    source._source_frame_index = 1
    source._frame_pump_position = 15.0 + 1.0 / 120.0
    source._frame_pump_thread = threading.current_thread()
    source._current_position = lambda: 100.0

    for _ in range(3):
        assert source._read_d3d11_frame() == "gpu-image"

    base = 15.0 + 1.0 / 120.0
    assert calls == pytest.approx([
        base + 2.5 / 120.0,
        base + 2.5 / 120.0,
        base + 3.5 / 120.0,
    ])


def test_native_media_decoder_is_retained_for_pause_resume():
    source = object.__new__(MediaCaptureSource)
    source._native_decoder = object()
    source._decode_backend = "ffmpeg-d3d11va-gpu"
    source._video_process = None
    assert source._has_open_video_decoder()
    source._native_decoder = None
    assert not source._has_open_video_decoder()


def test_media_audio_setup_uses_dac_callback_and_precise_seek_pts(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "linux")

    class Stream:
        time = 100.0
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)
        def start(self):
            pass
    class Worker:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=Stream, RawOutputStream=Stream))
    monkeypatch.setattr("capture.media._preferred_audio_output_device", lambda _sd: None)
    commands = []
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: commands.append(command) or SimpleNamespace())
    monkeypatch.setattr(threading, "Thread", Worker)
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_presentation_sync=True)
    source._ffmpeg = "ffmpeg"
    source._input_args = ["-i", "movie.mp4"]
    source._media_generation = 3
    source._close_audio = lambda: pytest.fail("audio setup unexpectedly failed")
    source._start_audio(12.345678912)
    assert commands[0][commands[0].index("-ss") + 1] == "12.345678912"
    assert "-re" not in commands[0]
    assert commands[0][commands[0].index("-af") + 1] == "aresample=async=1:first_pts=0"
    assert calls[0]["blocksize"] == 480
    assert callable(calls[0]["callback"])
    assert source._audio_started_at is None
    assert source._audio_sync.generation == 3


def test_media_audio_setup_uses_raw_stream_on_macos(monkeypatch):
    calls = []

    class Stream:
        time = 100.0

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def start(self):
            pass

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=Stream, RawOutputStream=Stream),
    )
    monkeypatch.setattr("capture.media._preferred_audio_output_device", lambda _sd: None)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: SimpleNamespace())
    monkeypatch.setattr(threading, "Thread", Worker)
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_presentation_sync=True)
    source._ffmpeg = "ffmpeg"
    source._input_args = ["-i", "movie.mp4"]
    source._media_generation = 3
    source._close_audio = lambda: pytest.fail("audio setup unexpectedly failed")

    source._start_audio(0.0)

    assert calls[0]["blocksize"] == 480
    assert "callback" not in calls[0]


def test_media_audio_reader_assembles_partial_samples_without_timestamp_loss():
    data = np.arange(960, dtype=np.float32).reshape(480, 2)
    class PartialPipe:
        def __init__(self):
            self.bytes = io.BytesIO(data.tobytes())
        def read(self, count):
            return self.bytes.read(min(count, 137))
    chunks = []
    source = object.__new__(MediaCaptureSource)
    source._audio_process = SimpleNamespace(stdout=PartialPipe())
    source._audio_stream = object()
    source._audio_stop = threading.Event()
    source._audio_sync = SimpleNamespace(enqueue=lambda samples, stop: chunks.append(samples) or True)
    source._audio_loop()
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], data)


def test_media_presentation_feedback_matches_generation_before_audio_clock():
    calls = []
    source = object.__new__(MediaCaptureSource)
    source._media_generation = 2
    source._presented_media_identity = None
    source._presentation_lock = threading.Lock()
    source._audio_sync = SimpleNamespace(observe_video=lambda *args: calls.append(args))
    source.playing = True
    source._playback_fps = 24.0
    source.observe_video_presentation(1, 30, 10.0, 100.0)
    assert calls == []
    source.observe_video_presentation(2, 0, 20.0, 101.0)
    source.observe_video_presentation(2, 0, 20.0, 102.0)
    assert calls == [(2, 0, 20.0, 101.0, 1.0 / 24.0)]
    source.playing = False
    source.observe_video_presentation(2, 5, 20.042, 102.0)
    assert source.position == 20.042
    assert len(calls) == 1


def test_audio_dac_mapping_does_not_inherit_delayed_callback_entry_time(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    class Stream:
        time = 100.0
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
        def start(self):
            pass
    class Worker:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=Stream, RawOutputStream=Stream))
    monkeypatch.setattr("capture.media._preferred_audio_output_device", lambda _sd: None)
    monkeypatch.setattr("capture.media.time.perf_counter", lambda: 200.0)
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_kw: SimpleNamespace())
    monkeypatch.setattr(threading, "Thread", Worker)
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_presentation_sync=True)
    source._ffmpeg = "ffmpeg"
    source._input_args = ["-i", "movie.mp4"]
    source._media_generation = 3
    source.volume, source.muted = 1.0, False
    source._close_audio = lambda: pytest.fail("audio setup unexpectedly failed")
    source._start_audio(10.0)
    source._audio_sync.observe_video(3, 0, 10.0, 200.0, 0.04)
    source._audio_sync.enqueue(np.ones((960, 2), dtype=np.float32), source._audio_stop)
    output = np.zeros((480, 2), dtype=np.float32)
    source._audio_stream.callback(output, 480,
        SimpleNamespace(outputBufferDacTime=100.01, currentTime=99.5),
        SimpleNamespace(output_underflow=False))
    assert np.all(output == 1.0)
    assert source._audio_sync.snapshot(200.01)["audio_pts"] == pytest.approx(10.01)


def test_native_decode_ahead_waits_only_on_its_import_stream_before_publishing(monkeypatch):
    from contextlib import contextmanager
    import torch

    calls = []
    stream = SimpleNamespace(synchronize=lambda: calls.append("complete_import"))
    @contextmanager
    def isolated_stream(selected):
        assert selected is stream
        calls.append("enter_import_stream")
        yield
        calls.append("leave_import_stream")
    monkeypatch.setattr(torch.cuda, "stream", isolated_stream)
    source = object.__new__(MediaCaptureSource)
    source._native_decoder = object()
    source._native_frame_importer = SimpleNamespace(_stream=stream)
    owned = object()
    borrowed = SimpleNamespace(clone=lambda: calls.append("clone_ring_image") or owned)
    source._read_video_frame = lambda: calls.append("decode") or borrowed
    assert source._read_buffered_video_frame() is owned
    assert calls == ["enter_import_stream", "decode", "clone_ring_image", "leave_import_stream", "complete_import"]


def test_native_decode_failure_keeps_software_fallback_without_tensor_clone(monkeypatch):
    from contextlib import nullcontext
    import torch

    source = object.__new__(MediaCaptureSource)
    source._native_frame_importer = SimpleNamespace(_stream=SimpleNamespace(synchronize=lambda: None))
    source._native_decoder = object()
    monkeypatch.setattr(torch.cuda, "stream", lambda *_args: nullcontext())
    host_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    def fallback():
        source._native_decoder = None
        return host_frame
    source._read_video_frame = fallback
    assert source._read_buffered_video_frame() is host_frame


def test_native_cadence_change_does_not_launch_unused_ffmpeg_pipe(monkeypatch):
    monkeypatch.setenv("D2S_MEDIA_CADENCE_ADAPTIVE", "1")
    source = object.__new__(MediaCaptureSource)
    source._source_fps = 120.0
    source.width, source.height = 3840, 1920
    source.config = SimpleNamespace(fps=60.0)
    source._playback_fps = 60.0
    source._frame_pump_decimation_step = 2
    source._adaptive_cadence_target = 30.0
    source._frame_pump_stop = threading.Event()
    source._decode_backend = "ffmpeg-d3d11va-gpu"
    source._native_decoder = object()
    source._frame_buffer = deque()
    source._frame_buffer_condition = threading.Condition()
    source._start_ffmpeg_video = lambda *_a, **_k: pytest.fail("native decoder must remain GPU resident")
    source._apply_media_playback_cadence_target()
    assert source._playback_fps == 30.0
    assert source._frame_pump_decimation_step == 4


def test_adaptive_cadence_drains_selected_frames_before_changing_divisor(monkeypatch):
    monkeypatch.setenv("D2S_MEDIA_CADENCE_ADAPTIVE", "1")
    source = object.__new__(MediaCaptureSource)
    source._source_fps = 120.0
    source.width, source.height = 3840, 1920
    source.config = SimpleNamespace(fps=60.0)
    source._playback_fps = 60.0
    source._frame_pump_decimation_step = 2
    source._adaptive_cadence_target = 30.0
    source._frame_pump_stop = threading.Event()
    source._frame_buffer = deque([(0.0, "selected", 0), (2 / 120, "selected-next", 2)])
    source._frame_buffer_condition = threading.Condition()
    worker = threading.Thread(target=source._apply_media_playback_cadence_target)
    worker.start()
    time.sleep(0.02)
    with source._frame_buffer_condition:
        assert source._playback_fps == 60.0
        assert [item[2] for item in source._frame_buffer] == [0, 2]
        selected = [source._frame_buffer.popleft(), source._frame_buffer.popleft()]
        source._frame_buffer_condition.notify_all()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert [item[2] for item in selected] == [0, 2]
    assert source._playback_fps == 30.0
    assert source._frame_pump_decimation_step == 4


def test_nvdec_discards_intermediate_surface_batch_before_next_fetch():
    import weakref
    source = object.__new__(MediaCaptureSource)
    discarded = []
    calls = []
    class Surface:
        pass
    class Decoder:
        def get_batch_frames(self, count):
            calls.append(count)
            if len(calls) == 1:
                batch = [Surface() for _ in range(count)]
                discarded.extend(weakref.ref(frame) for frame in batch)
                return batch
            assert all(reference() is None for reference in discarded)
            return []
    source._nvdec_decoder = Decoder()
    source._nvdec_torch = object()
    source._nvdec_buffer_size = 4
    assert source._decode_nvdec_frames(8) is None
    assert calls == [4, 4]


def test_media_preroll_does_not_advance_source_until_first_successful_presentation(monkeypatch):
    monkeypatch.setenv("D2S_MEDIA_CADENCE_ADAPTIVE", "1")
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_presentation_sync=True, fps=60.0)
    source._source_fps = 120.0
    source.width, source.height = 3840, 1920
    source._playback_fps = 60.0
    source._media_generation = 4
    source._frame_pump_decimation_step = 2
    source._presentation_clock_pending = True
    source._presentation_first_frame_taken = False
    source._media_cadence_calibrated = False
    source._presentation_lock = threading.Lock()
    source._frame_buffer_condition = threading.Condition()
    source._frame_buffer = deque([(300.0, "first", 0), (300.017, "next", 2)])
    source._frame_pump_thread = SimpleNamespace(is_alive=lambda: True)
    source._frame_pump_done = False
    source._frame_pump_error = ""
    source._position_anchor_time = None
    source._position_anchor = source.position = 300.0
    source.duration = 346.366
    source.playing = source._ready = True
    source._image = False
    source._browser_mode = False
    source._presented_media_identity = None
    source._audio_sync = None
    source._read_control = lambda: None
    source._publish_state = lambda **_kwargs: None
    source._take_buffered_frame = lambda *_a, **_kw: source._frame_buffer.popleft()
    now = [100.0]
    monkeypatch.setattr("capture.media.time.perf_counter", lambda: now[0])

    assert source.grab()[0] == "first"
    assert source._current_source_metadata["media_presentation_clock_pending"] is True
    now[0] = 110.0  # Engine/WSI initialization must not start the movie clock.
    assert source._current_position() == 300.0
    assert source.grab()[0] is None
    assert [item[2] for item in source._frame_buffer] == [2]
    source.calibrate_media_cadence(27.0)
    assert source._playback_fps == 30.0
    assert source._frame_pump_decimation_step == 4
    assert source._media_cadence_calibrated
    source.observe_video_presentation(4, 0, 300.0, 110.0)
    assert not source._presentation_clock_pending
    now[0] = 112.0
    assert source._current_position() == 302.0
    source.observe_video_presentation(4, 0, 300.0, 112.0)
    assert source._position_anchor_time == 110.0  # Reusing the image cannot rebase time.


def test_media_preroll_uses_hardware_cost_and_retains_explicit_ceiling(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(fps=120.0)
    source._source_fps = 120.0
    source.width, source.height = 3840, 1920
    source._presentation_clock_pending = True
    source._media_cadence_calibrated = False
    source._frame_buffer_condition = threading.Condition()
    monkeypatch.setenv("D2S_MEDIA_CADENCE_ADAPTIVE", "1")
    monkeypatch.delenv("D2S_MEDIA_DECIMATION_TARGET_FPS", raising=False)
    reset_media_playback_cadence_target()
    source.calibrate_media_cadence(6.0)
    assert source._playback_fps == 120.0
    source._media_cadence_calibrated = False
    source.calibrate_media_cadence(27.0)
    assert source._playback_fps == 30.0
    source._media_cadence_calibrated = False
    source.calibrate_media_cadence(60.0)
    assert source._playback_fps == 15.0
    source._media_cadence_calibrated = False
    monkeypatch.setenv("D2S_MEDIA_DECIMATION_TARGET_FPS", "24")
    source.calibrate_media_cadence(6.0)
    assert source._playback_fps == 24.0
    monkeypatch.delenv("D2S_MEDIA_DECIMATION_TARGET_FPS")
    source._media_cadence_calibrated = False
    source.calibrate_media_cadence(23.95)
    assert source._playback_fps == 40.0


def test_media_preroll_keeps_requested_rate_when_adaptive_cadence_is_disabled(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(fps=60.0)
    source._source_fps = 120.0
    source.width, source.height = 3840, 1920
    source._playback_fps = 60.0
    source._frame_pump_decimation_step = 2
    source._presentation_clock_pending = True
    source._media_cadence_calibrated = False
    source._frame_buffer_condition = threading.Condition()
    monkeypatch.setenv("D2S_MEDIA_CADENCE_ADAPTIVE", "0")

    source.calibrate_media_cadence(60.0)

    assert source._playback_fps == 60.0
    assert source._frame_pump_decimation_step == 2
    assert source._last_source_frame_duration == pytest.approx(1.0 / 60.0)
    assert source._media_prepare_lead_s == pytest.approx(1.0 / 60.0 + 0.06)


def test_media_runtime_target_is_ignored_when_adaptive_cadence_is_disabled(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source._source_fps = 120.0
    source.width, source.height = 3840, 1920
    source.config = SimpleNamespace(fps=60.0)
    source._playback_fps = 60.0
    source._frame_pump_decimation_step = 2
    source._frame_pump_stop = threading.Event()
    source._frame_buffer = deque()
    source._frame_buffer_condition = threading.Condition()
    monkeypatch.setenv("D2S_MEDIA_CADENCE_ADAPTIVE", "0")

    try:
        set_media_playback_cadence_target(24.0)
        source._apply_media_playback_cadence_target()
    finally:
        reset_media_playback_cadence_target()

    assert source._playback_fps == 60.0
    assert source._frame_pump_decimation_step == 2


def test_media_preparation_lead_uses_fresh_measured_latency_not_repeated_frame_age():
    source = object.__new__(MediaCaptureSource)
    source.playing = True
    source._media_generation = 1
    source._source_fps = 120.0
    source._playback_fps = 30.0
    source._media_prepare_lead_s = 0.0
    source._media_cadence_controller = SimpleNamespace(observe=lambda **_kw: None)
    source._media_cadence_generation = 1
    source.runtime_metrics = lambda: {"buffer_depth": 8}
    metadata = {"media_generation": 1, "source_frame_index": 0, "source_pts": 10.0,
                "media_playback_fps": 30.0, "capture_poll_returned_at": 99.0}
    source.observe_media_timing(metadata, 100.0)  # Initial preroll is not per-frame latency.
    assert source._media_prepare_lead_s == 0.0
    metadata.update(source_frame_index=4, source_pts=10.033333, capture_poll_returned_at=100.0)
    source.observe_media_timing(metadata, 100.045)
    assert source._media_prepare_lead_s == pytest.approx(0.045)
    source.observe_media_timing(metadata, 100.2)  # XR reuses the same displayed source frame.
    assert source._media_prepare_lead_s == pytest.approx(0.045)


def test_decode_ahead_eof_keeps_audio_and_last_image_until_source_duration(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source._ready = source.playing = True
    source._image = source._browser_mode = False
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_thread = SimpleNamespace(is_alive=lambda: True)
    source._frame_buffer = deque()
    source._frame_pump_done = True
    source._frame_pump_error = ""
    source._position_anchor_time = 100.0
    source._position_anchor = 0.0
    source.duration = 10.0
    source.width, source.height = 1920, 1080
    source._media_prepare_lead_s = 0.1
    source._read_control = lambda: None
    source._publish_state = lambda **_kwargs: None
    source._take_buffered_frame = lambda *_a, **_kw: None
    finished = []
    source._finish_video_stream = lambda: finished.append(True) or "last-image"
    now = [109.95]
    monkeypatch.setattr("capture.media.time.perf_counter", lambda: now[0])
    assert source.grab()[0] is None
    assert not finished
    now[0] = 110.0
    assert source.grab()[0] == "last-image"
    assert finished == [True]


def test_resume_epoch_ignores_old_queued_frames_before_first_new_capture():
    source = object.__new__(MediaCaptureSource)
    source.playing = True
    source._media_generation = 1
    source._presentation_lock = threading.Lock()
    source._frame_buffer_condition = threading.Condition()
    source._presentation_clock_pending = True
    source._presentation_start_index = 105
    source._presented_media_identity = (1, 100, 10.0, 100.0)
    source._audio_sync = None
    source._position_anchor = source.position = 10.5
    source._position_anchor_time = None
    source.observe_video_presentation(1, 103, 10.3, 105.0)
    assert source._presentation_clock_pending
    assert source._position_anchor_time is None
    assert source._presented_media_identity[1] == 100
    source.observe_video_presentation(1, 105, 10.5, 105.1)
    assert not source._presentation_clock_pending
    assert source._position_anchor_time == 105.1
    assert source._position_anchor == 10.5


def test_presentation_synced_decimation_uses_pts_not_a_second_emit_phase(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_presentation_sync=True)
    source._playback_fps = 24.0
    source._frame_pump_decimation_step = 5
    source._last_decimated_emit_at = 100.0  # A recent late result must not move the source clock.
    source._last_source_frame_duration = 1.0 / 24.0
    source._frame_buffer = deque([(10.0, "first", 0), (10.0 + 1 / 24, "next", 5)])
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_done = True
    ticks = iter(100.002 + i * 0.002 for i in range(100))
    monkeypatch.setattr("capture.media.time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    assert source._take_buffered_frame(10.01) == (10.0, "first", 0)
    # Source time is already due after a stall; hand the next selected frame
    # to preparation without an additional relative wait or any discard.
    assert source._take_buffered_frame(10.05) == (10.0 + 1 / 24, "next", 5)
    assert source._last_decimated_emit_at == 100.0


def test_presentation_sync_drops_stale_frames_to_keep_wall_clock(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_presentation_sync=True)
    source._playback_fps = 60.0
    source._source_fps = 60.0
    source._last_source_frame_duration = 1.0 / 60.0
    source._frame_pump_decimation_step = 1
    source._frame_buffer = deque(
        [(10.00, "stale", 0), (10.50, "due", 30), (11.00, "future", 60)]
    )
    source._frame_buffer_condition = threading.Condition()
    source._frame_pump_done = False
    source._frame_buffer_stale_drops = 0
    source._frame_buffer_drop_log_at = 0.0
    monkeypatch.setattr("capture.media.time.perf_counter", lambda: 100.0)

    assert source._take_buffered_frame(10.75) == (10.50, "due", 30)
    assert source._frame_buffer_stale_drops == 1
    assert list(source._frame_buffer) == [(11.00, "future", 60)]


def test_pause_resume_preserves_the_remaining_source_frame_interval(monkeypatch):
    source = object.__new__(MediaCaptureSource)
    source.config = SimpleNamespace(media_presentation_sync=True)
    source._media_generation = 1
    source.playing = True
    source._position_anchor = 10.0
    source._position_anchor_time = 100.0
    source.duration = 30.0
    source._presentation_lock = threading.Lock()
    source._frame_buffer_condition = threading.Condition()
    source._frame_buffer = deque([(10.05, "prepared", 105)])
    source._presented_media_identity = (1, 100, 10.0, 100.0)
    source._audio_sync = None
    source._close_audio = lambda: None
    source._stop_frame_pump = lambda: None
    source._start_frame_pump = lambda: None
    clock = [100.02]
    monkeypatch.setattr("capture.media.time", SimpleNamespace(perf_counter=lambda: clock[0]))
    source._pause_playback()
    assert source.position == 10.0
    assert source._paused_clock_pts == pytest.approx(10.02)
    clock[0] = 104.02
    source._resume_playback(source.position)
    snapshot = source.media_presentation_clock()
    assert snapshot["anchor_at"] == 104.02
    assert snapshot["anchor_pts"] == pytest.approx(10.02)
    source.observe_video_presentation(1, 105, 10.05, 104.05)
    assert source._position_anchor_time == 104.02
    assert source._position_anchor == pytest.approx(10.02)
    clock[0] = 105.0
    assert source._current_position() == pytest.approx(11.0)
