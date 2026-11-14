import io
import json
import os
import sys
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
    _probe_media,
    _resolve_amf_output_size,
    _video_filter,
    _video_pixel_format,
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


def test_video_decode_falls_back_to_ffmpeg_for_unsupported_hardware():
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
    source = _source_for_decode_start()
    source._amf_decoder = None
    starts = []
    source._start_ffmpeg_video = lambda position, *, realtime: starts.append(
        (position, realtime)
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0),
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
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0),
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
    assert source._decode_backend == "nvdec"


def test_nvdec_unsupported_preflight_falls_back_before_threaded_decoder(monkeypatch):
    source = _source_for_decode_start()
    starts = []
    threaded_starts = []
    source._start_ffmpeg_video = lambda position, *, realtime: starts.append(
        (position, realtime)
    )
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0),
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
def test_hdr_ffmpeg_path_preserves_transfer_for_device_preprocess(transfer, primaries, frame_format):
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


def test_buffered_playback_uses_wall_clock_target_and_drops_stale_frames():
    source = object.__new__(MediaCaptureSource)
    source._playback_fps = 60.0
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

    assert item[1] == "frame-2"
    assert item[2] == 2
    assert list(source._frame_buffer) == []


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
