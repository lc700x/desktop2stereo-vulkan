from __future__ import annotations

import ssl
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from streaming import moonlight_output
from streaming.moonlight_bridge import MoonlightClientConfig
from streaming.moonlight_output import MoonlightSbsOutput, read_sunshine_base_port
from streaming.moonlight_paths import (
    moonlight_state_directory,
    sunshine_executable,
    sunshine_working_directory,
)


def _sunshine_config_output(tmp_path):
    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=(0, 0, 1920, 1080, 1920, 1080),
        device="cuda:0",
    )
    output._state_dir = tmp_path
    output._sunshine_config = tmp_path / "sunshine.conf"
    output._sunshine_apps = tmp_path / "apps.json"
    output._sunshine_credentials = tmp_path / "web_credentials"
    output._sunshine_account_file = tmp_path / "web_account.json"
    output._sunshine_state = tmp_path / "sunshine_state.json"
    output._sunshine_pki_dir = tmp_path / "sunshine-pki"
    output._sunshine_private_key = output._sunshine_pki_dir / "cakey.pem"
    output._sunshine_certificate = output._sunshine_pki_dir / "cacert.pem"
    return output


def test_sunshine_api_uses_truststore_inner_context_for_pinned_tls(
    monkeypatch, tmp_path
):
    certificate_der = b"pinned Sunshine certificate"
    certificate_pem = ssl.DER_cert_to_PEM_cert(certificate_der)
    certificate_path = tmp_path / "sunshine-pki" / "cacert.pem"
    certificate_path.parent.mkdir()
    certificate_path.write_text(certificate_pem, encoding="ascii")
    raw_context = type("RawContext", (), {})()

    class TruststoreContext:
        def __init__(self, _protocol):
            object.__setattr__(self, "_ctx", raw_context)

        @property
        def check_hostname(self):
            return self._ctx.check_hostname

        @check_hostname.setter
        def check_hostname(self, value):
            self._ctx.check_hostname = value

        @property
        def verify_mode(self):
            return self._ctx.verify_mode

        @verify_mode.setter
        def verify_mode(self, value):
            self._ctx.verify_mode = value

    class FakeSocket:
        def getpeercert(self, *, binary_form):
            assert binary_form is True
            return certificate_der

    class FakeResponse:
        status = 200

        @staticmethod
        def read():
            return b'{"status": true}'

    calls = {}

    class FakeConnection:
        def __init__(self, host, port, *, context, timeout):
            calls.update(host=host, port=port, context=context, timeout=timeout)
            self.sock = None

        def connect(self):
            self.sock = FakeSocket()

        def request(self, method, endpoint, *, body, headers):
            calls.update(method=method, endpoint=endpoint, body=body, headers=headers)

        @staticmethod
        def getresponse():
            return FakeResponse()

        @staticmethod
        def close():
            calls["closed"] = True

    monkeypatch.setattr(moonlight_output.ssl, "SSLContext", TruststoreContext)
    monkeypatch.setattr(moonlight_output.http.client, "HTTPSConnection", FakeConnection)

    result = moonlight_output._request_sunshine_api(
        state_dir=tmp_path,
        base_port=47989,
        username="d2s",
        password="secret",
        method="GET",
        endpoint="/api/config",
    )

    assert result == {"status": True}
    assert calls["context"] is raw_context
    assert raw_context.check_hostname is False
    assert raw_context.verify_mode == ssl.CERT_NONE
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 47990
    assert calls["method"] == "GET"
    assert calls["endpoint"] == "/api/config"
    assert calls["closed"] is True


class _OccupiedSocket:
    def __init__(self, occupied_ports):
        self.occupied_ports = occupied_ports

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def bind(self, address):
        if address[1] in self.occupied_ports:
            raise OSError("address already in use")


def test_moonlight_selects_alternate_port_family_when_default_is_busy():
    with patch(
        "streaming.moonlight_output.socket.socket",
        side_effect=lambda _family, _socket_type: _OccupiedSocket({47984}),
    ):
        assert MoonlightSbsOutput._select_sunshine_port(47989) == 48089


def test_moonlight_state_paths_and_bundled_macos_helper():
    assert moonlight_state_directory(
        platform_name="darwin", environ={}
    ) == __import__("pathlib").Path.home() / "Library" / "Application Support" / "Desktop2Stereo" / "Moonlight"
    assert moonlight_state_directory(
        platform_name="win32", environ={"LOCALAPPDATA": "/tmp/local"}
    ) == __import__("pathlib").Path("/tmp/local/Desktop2Stereo/Moonlight")
    app_exe = sunshine_executable("/bundle/server", platform_name="darwin", environ={})
    assert app_exe.as_posix().endswith(
        "/bundle/server/macos/Sunshine.app/Contents/MacOS/Sunshine"
    )
    assert sunshine_working_directory(app_exe).as_posix().endswith(
        "/bundle/server/macos/Sunshine.app/Contents/Resources"
    )


def test_macos_moonlight_encodes_native_coreml_metal_frame():
    class _NativeFrame:
        released = False

        def release(self):
            self.released = True

    class _Encoder:
        def encode_native_frame(self, frame, **kwargs):
            assert frame is native_frame
            assert kwargs["frame_index"] == 0
            assert kwargs["force_keyframe"] is True
            encoded.set()
            return {"data": b"annex-b", "frame_index": 0, "idr": True}

    class _Bridge:
        connected = True
        connection_generation = 7

        def __init__(self):
            self.sent = []

        def send_access_unit(self, *args, **kwargs):
            self.sent.append((args, kwargs))
            sent.set()
            return True

        def close(self):
            self.connected = False

    output = MoonlightSbsOutput(
        base_dir=".",
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    native_frame = _NativeFrame()
    encoded = threading.Event()
    sent = threading.Event()
    bridge = _Bridge()
    output._bridge = bridge
    output._encoder = _Encoder()
    output._encoder_config = MoonlightClientConfig(
        1920, 1080, 60.0, 20_000, "h264", 1920, 1080,
        0, 0, 1920, 1080, 0,
    )
    output._force_idr.set()
    result = type("RuntimeResult", (), {
        "native_stream_frame": native_frame,
        "output_format": "half_sbs",
        "sbs": object(),
    })()
    timestamp = 12.5
    output._native_session_ready = True

    try:
        with patch("streaming.moonlight_output.sys.platform", "darwin"):
            output.submit_runtime_result(result, timestamp)

        assert result.native_stream_frame is None
        assert encoded.wait(1.0)
        assert sent.wait(1.0)
        assert native_frame.released
        args, kwargs = bridge.sent[0]
        assert args == (0, 12_500_000_000, b"annex-b")
        assert kwargs == {"idr": True, "client_generation": 7}
    finally:
        output.close()


def test_macos_moonlight_paces_native_frames_to_negotiated_fps():
    class _NativeFrame:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    class _Encoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    class _Bridge:
        connected = True
        connection_generation = 4

        def close(self):
            self.connected = False

    output = MoonlightSbsOutput(
        base_dir=".",
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    output._bridge = _Bridge()
    frames = [_NativeFrame() for _ in range(5)]
    submissions = []
    clock = [100.0]

    def submit(frame, timestamp):
        result = type("RuntimeResult", (), {
            "native_stream_frame": frame,
            "output_format": "half_sbs",
        })()
        output.submit_runtime_result(result, timestamp)

    try:
        with (
            patch("streaming.moonlight_output.sys.platform", "darwin"),
            patch(
                "streaming.moonlight_output.time.perf_counter",
                side_effect=lambda: clock[0],
            ),
            patch("streaming.videotoolbox_encoder.VideoToolboxEncoder", _Encoder),
        ):
            output._configure_encoder(
                MoonlightClientConfig(
                    1920, 1080, 30.0, 15_000, "hevc", 1920, 1080,
                    0, 0, 1920, 1080, 0,
                )
            )
            output._ensure_native_encoder_worker = lambda: None
            output._enqueue_native_frame = (
                lambda submission: submissions.append(submission) or True
            )
            submit(frames[0], 1.0)
            submit(frames[1], 1.1)
            output._request_keyframe()
            submit(frames[2], 1.2)
            first_batch = list(submissions)
            output._configure_encoder(
                MoonlightClientConfig(
                    1920, 1080, 60.0, 15_000, "hevc", 1920, 1080,
                    0, 0, 1920, 1080, 0,
                )
            )
            submissions.clear()
            clock[0] += 1.0 / 60.0
            submit(frames[3], 1.3)
            clock[0] += 1.0 / 60.0
            submit(frames[4], 1.4)

        assert [submission.frame_index for submission in first_batch] == [0, 1]
        assert first_batch[-1].force_keyframe
        assert [submission.frame_index for submission in submissions] == [0, 1]
        assert submissions[0].force_keyframe
        assert not submissions[1].force_keyframe
        assert not frames[0].released
        assert frames[1].released
        assert not frames[2].released
        assert not frames[3].released
        assert not frames[4].released
    finally:
        output.close()


def test_macos_moonlight_drops_stale_native_frames_under_backpressure():
    class _NativeFrame:
        def __init__(self, marker):
            self.marker = marker
            self.released = threading.Event()

        def release(self):
            self.released.set()

    class _Encoder:
        def __init__(self):
            self.started = threading.Event()
            self.allow_first = threading.Event()
            self.frames = []

        def encode_native_frame(self, frame, **kwargs):
            self.frames.append(frame.marker)
            if frame.marker == 1:
                self.started.set()
                assert self.allow_first.wait(2.0)
            return {
                "data": bytes((frame.marker,)),
                "frame_index": kwargs["frame_index"],
                "idr": kwargs["force_keyframe"],
            }

        def close(self):
            pass

    class _Bridge:
        connected = True
        connection_generation = 11

        def __init__(self):
            self.sent = []
            self.sent_latest = threading.Event()

        def send_access_unit(self, _index, _timestamp, payload, **_kwargs):
            self.sent.append(payload[0])
            if payload[0] == 3:
                self.sent_latest.set()
            return True

        def close(self):
            self.connected = False

    output = MoonlightSbsOutput(
        base_dir=".",
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    encoder = _Encoder()
    bridge = _Bridge()
    output._bridge = bridge
    output._encoder = encoder
    output._encoder_config = MoonlightClientConfig(
        1920, 1080, 60.0, 20_000, "h264", 1920, 1080,
        0, 0, 1920, 1080, 0,
    )
    output._native_session_ready = True
    # This test exercises the one-slot backpressure queue independently of
    # negotiated-rate pacing.
    output._native_target_fps = 1_000_000.0
    frames = [_NativeFrame(marker) for marker in (1, 2, 3)]

    def submit(frame, capture_timestamp):
        result = type("RuntimeResult", (), {
            "native_stream_frame": frame,
            "output_format": "half_sbs",
        })()
        output.submit_runtime_result(result, capture_timestamp)
        assert result.native_stream_frame is None

    try:
        with patch("streaming.moonlight_output.sys.platform", "darwin"):
            submit(frames[0], 1.0)
            assert encoder.started.wait(1.0)
            submit(frames[1], 2.0)
            submit(frames[2], 3.0)
            assert frames[1].released.wait(1.0)
            # The encoder is deliberately blocked on frame 1; submissions still
            # return immediately and the one waiting slot retains only frame 3.
            encoder.allow_first.set()
            assert bridge.sent_latest.wait(1.0)

        assert encoder.frames == [1, 3]
        assert bridge.sent == [1, 3]
        assert all(frame.released.wait(1.0) for frame in frames)
    finally:
        encoder.allow_first.set()
        output.close()


def test_macos_moonlight_close_releases_waiting_native_frame():
    class _NativeFrame:
        def __init__(self):
            self.released = threading.Event()

        def release(self):
            self.released.set()

    class _Encoder:
        def __init__(self):
            self.started = threading.Event()
            self.allow_encode = threading.Event()

        def encode_native_frame(self, _frame, **kwargs):
            self.started.set()
            assert self.allow_encode.wait(2.0)
            return {
                "data": b"encoded",
                "frame_index": kwargs["frame_index"],
                "idr": kwargs["force_keyframe"],
            }

        def close(self):
            pass

    class _Bridge:
        connected = True
        connection_generation = 3

        def close(self):
            self.connected = False

    output = MoonlightSbsOutput(
        base_dir=".",
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    encoder = _Encoder()
    output._bridge = _Bridge()
    output._encoder = encoder
    output._encoder_config = MoonlightClientConfig(
        1920, 1080, 60.0, 20_000, "h264", 1920, 1080,
        0, 0, 1920, 1080, 0,
    )
    output._native_session_ready = True
    active, waiting = _NativeFrame(), _NativeFrame()

    def submit(frame):
        result = type("RuntimeResult", (), {"native_stream_frame": frame})()
        output.submit_runtime_result(result, 1.0)

    closer = threading.Thread(target=output.close, daemon=True)
    try:
        with patch("streaming.moonlight_output.sys.platform", "darwin"):
            submit(active)
            assert encoder.started.wait(1.0)
            submit(waiting)
        closer.start()
        assert waiting.released.wait(1.0)
        encoder.allow_encode.set()
        closer.join(timeout=2.0)
        assert not closer.is_alive()
        assert active.released.wait(1.0)
    finally:
        encoder.allow_encode.set()
        if closer.ident is not None:
            closer.join(timeout=3.0)
        else:
            output.close()


def test_macos_moonlight_without_a_client_does_not_require_native_frame():
    output = MoonlightSbsOutput(
        base_dir=".",
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    result = type("RuntimeResult", (), {"native_stream_frame": None})()
    with patch("streaming.moonlight_output.sys.platform", "darwin"):
        output.submit_runtime_result(result, 0.0)


def test_macos_moonlight_explains_missing_native_metal_path():
    class _Bridge:
        connected = True

    output = MoonlightSbsOutput(
        base_dir=".",
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    output._bridge = _Bridge()
    result = type("RuntimeResult", (), {"native_stream_frame": None})()
    with patch("streaming.moonlight_output.sys.platform", "darwin"):
        try:
            output.submit_runtime_result(result, 0.0)
        except RuntimeError as exc:
            assert "Enable CoreML" in str(exc)
            assert "Screen Recording" in str(exc)
        else:
            raise AssertionError("missing native path should give a setup message")


def test_macos_moonlight_accepts_video_toolbox_for_a_client():
    class _Encoder:
        def __init__(self, width, height, fps, bitrate, *, hevc, **kwargs):
            assert (width, height, fps, bitrate, hevc) == (
                1920, 1080, 60, 18_000_000, False
            )
            assert kwargs == {
                "fit_mode": "contain",
                "display_mode": "Half-SBS",
                "input_size": (1920, 1080),
            }

        def close(self):
            pass

    output = MoonlightSbsOutput(
        base_dir=".",
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    config = MoonlightClientConfig(
        1920, 1080, 60.0, 18_000, "h264", 1920, 1080,
        0, 0, 1920, 1080, 0,
    )

    with (
        patch("streaming.moonlight_output.sys.platform", "darwin"),
        patch("streaming.videotoolbox_encoder.VideoToolboxEncoder", _Encoder),
    ):
        response = output._configure_encoder(config)

    assert response == {"accepted_width": 1920, "accepted_height": 1080}
    assert isinstance(output._encoder, _Encoder)


def test_macos_moonlight_renegotiation_recreates_and_closes_encoder():
    class _Encoder:
        instances = []

        def __init__(self, width, height, fps, bitrate, *, hevc, **_kwargs):
            self.settings = (width, height, fps, bitrate, hevc)
            self.closed = False
            self.instances.append(self)

        def close(self):
            self.closed = True

    class _Bridge:
        connected = True

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    output = MoonlightSbsOutput(
        base_dir=".",
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    bridge = _Bridge()
    output._bridge = bridge
    first = MoonlightClientConfig(
        1920, 1080, 60.0, 18_000, "h264", 1920, 1080,
        0, 0, 1920, 1080, 0,
    )
    renegotiated = MoonlightClientConfig(
        1280, 720, 30.0, 10_000, "hevc", 1280, 720,
        0, 0, 1280, 720, 0,
    )

    with (
        patch("streaming.moonlight_output.sys.platform", "darwin"),
        patch("streaming.videotoolbox_encoder.VideoToolboxEncoder", _Encoder),
    ):
        output._configure_encoder(first)
        output._configure_encoder(renegotiated)
        assert _Encoder.instances[0].closed
        assert _Encoder.instances[1].settings == (1280, 720, 30, 10_000_000, True)
        output.close()

    assert _Encoder.instances[1].closed
    assert bridge.closed


def test_videotoolbox_native_pack_composes_and_fits_each_eye_on_metal():
    import ctypes

    from streaming.videotoolbox_encoder import VideoToolboxEncoder

    class _NativeFrame:
        source_size = (1920, 1080)

        def __init__(self):
            self.packed = None
            self.warp_config = {"depth_strength": 0.4}

        def configure_warp(self, **values):
            self.warp_config = dict(values)

        def pack_metal(self, destination, destination_size, output_size, output_format):
            self.packed = (destination, destination_size, output_size, output_format)

    class _Library:
        def d2s_videotoolbox_source_buffer(self, _handle, width, height, _error, _size):
            self.source_size = (width, height)
            return 1234

        def d2s_videotoolbox_encode(
            self, _handle, _frame_index, _force, region_count, source_bgra, regions,
            output, output_size, keyframe, _error, _error_size,
        ):
            self.source_bgra = source_bgra
            self.regions = tuple(regions[: region_count * 8])
            self.encoded = (ctypes.c_uint8 * 7)(0, 0, 0, 1, 0x65, 0x88, 0x84)
            ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)))[0] = (
                ctypes.cast(self.encoded, ctypes.POINTER(ctypes.c_uint8))
            )
            ctypes.cast(output_size, ctypes.POINTER(ctypes.c_size_t))[0] = len(self.encoded)
            ctypes.cast(keyframe, ctypes.POINTER(ctypes.c_int))[0] = 1
            return 0

    encoder = VideoToolboxEncoder.__new__(VideoToolboxEncoder)
    encoder.width = 1280
    encoder.height = 720
    encoder.fit_mode = "contain"
    encoder.display_mode = "Half-SBS"
    encoder.input_size = (1920, 1080)
    encoder._handle = 1
    encoder._library = _Library()
    native_frame = _NativeFrame()

    access_unit = encoder.encode_native_frame(
        native_frame,
        output_format="half_sbs",
        frame_index=7,
        force_keyframe=True,
    )

    assert encoder._library.source_size == (1920, 1080)
    assert native_frame.packed == (1234, 1920 * 1080 * 4, (1920, 1080), "half_sbs")
    assert native_frame.warp_config == {
        "depth_strength": 0.4,
        "half_sbs_single_sample": 1,
        "stream_bgra": 1,
    }
    assert encoder._library.source_bgra == 1
    assert encoder._library.regions == (
        0, 0, 960, 1080, 0, 0, 640, 720,
        960, 0, 1920, 1080, 640, 0, 1280, 720,
    )
    assert access_unit == {
        "data": b"\x00\x00\x00\x01\x65\x88\x84",
        "frame_index": 7,
        "idr": True,
    }


def test_macos_sunshine_launch_uses_app_resources_and_shared_state(tmp_path):
    app_resources = tmp_path / "Sunshine.app" / "Contents" / "Resources"
    app_binary = app_resources.parent / "MacOS" / "Sunshine"
    app_binary.parent.mkdir(parents=True)
    app_binary.touch()

    class _Bridge:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def start(self):
            return "127.0.0.1", 43821

        def close(self):
            self.closed = True

    class _Process:
        stdout = None

        def poll(self):
            return 0

    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="mps",
    )
    output._sunshine_exe = app_binary
    process = _Process()

    with (
        patch.object(output, "_select_sunshine_port", return_value=48089),
        patch.object(output, "_ensure_service_files"),
        patch.object(output, "_ensure_sunshine_account"),
        patch.object(output, "_start_pairing_request_watcher"),
        patch.object(output, "_sunshine_endpoint_in_use", return_value=False),
        patch("streaming.moonlight_output.MoonlightBridgeServer", _Bridge),
        patch("streaming.moonlight_output.subprocess.Popen", return_value=process) as popen,
        patch("streaming.moonlight_output.sys.platform", "darwin"),
    ):
        output.start()

    args, kwargs = popen.call_args
    assert args[0][0] == str(app_binary)
    assert kwargs["cwd"] == str(app_resources)
    assert kwargs["env"]["D2S_MOONLIGHT_SOURCE_WIDTH"] == "1920"
    output.close()


def test_moonlight_falls_back_when_another_sunshine_owns_default_port(
    monkeypatch, tmp_path, capsys
):
    output = _sunshine_config_output(tmp_path)
    sunshine_exe = tmp_path / "sunshine.exe"
    sunshine_exe.touch()
    output._sunshine_exe = sunshine_exe
    monkeypatch.setattr(output, "_sunshine_endpoint_in_use", lambda: True)
    monkeypatch.setattr(
        MoonlightSbsOutput,
        "_port_family_available",
        classmethod(lambda _cls, port: port == 48089),
    )
    monkeypatch.setattr(output, "_ensure_sunshine_account", lambda: None)
    monkeypatch.setattr(output, "_read_sunshine_log", lambda: None)
    monkeypatch.setattr(output, "_start_pairing_request_watcher", lambda: None)

    class FakeBridge:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            return "127.0.0.1", 12345

    monkeypatch.setattr(moonlight_output, "MoonlightBridgeServer", FakeBridge)
    monkeypatch.setattr(
        moonlight_output,
        "sunshine_working_directory",
        lambda _path: tmp_path,
    )
    monkeypatch.setattr(
        moonlight_output.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=None),
    )

    output.start()

    assert output._sunshine_port == 48089
    assert read_sunshine_base_port(output._sunshine_config) == 48089
    assert "selecting a compatible free port family" in capsys.readouterr().out


def test_sunshine_service_config_keeps_custom_port_and_host_name(tmp_path):
    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=(0, 0, 1920, 1080, 1920, 1080),
        device="cuda:0",
    )
    output._state_dir = tmp_path
    output._sunshine_config = tmp_path / "sunshine.conf"
    output._sunshine_apps = tmp_path / "apps.json"
    output._sunshine_credentials = tmp_path / "web_credentials"
    output._sunshine_account_file = tmp_path / "web_account.json"
    output._sunshine_state = tmp_path / "sunshine_state.json"
    output._sunshine_pki_dir = tmp_path / "credentials"
    output._sunshine_private_key = output._sunshine_pki_dir / "cakey.pem"
    output._sunshine_certificate = output._sunshine_pki_dir / "cacert.pem"
    output._sunshine_config.write_text(
        "port = 47989\nupnp = disabled\nsunshine_name = Old Host\n",
        encoding="utf-8",
    )

    output._ensure_service_files(48089)

    config = output._sunshine_config.read_text(encoding="utf-8")
    assert "port = 48089" in config
    assert "sunshine_name = D2S Sunshine" in config
    assert "upnp = disabled" in config
    assert read_sunshine_base_port(output._sunshine_config) == 48089
    output._sunshine_port = 48089
    assert output.pairing_url == "https://localhost:48090"


def test_macos_sunshine_config_leaves_audio_sink_blank_for_system_audio_tap(tmp_path):
    output = _sunshine_config_output(tmp_path)

    with patch("streaming.moonlight_output.sys.platform", "darwin"):
        output._ensure_service_files(48089)

    config = output._sunshine_config.read_text(encoding="utf-8")
    assert "audio_sink =\n" in config
    assert "audio_sink = default_audio_sink" not in config


def test_macos_sunshine_config_migrates_legacy_default_audio_sink(tmp_path):
    output = _sunshine_config_output(tmp_path)
    output._sunshine_config.write_text(
        "port = 47989\naudio_sink = default_audio_sink\n", encoding="utf-8"
    )

    with patch("streaming.moonlight_output.sys.platform", "darwin"):
        output._ensure_service_files(48089)

    config = output._sunshine_config.read_text(encoding="utf-8")
    assert "audio_sink =\n" in config
    assert "default_audio_sink" not in config


def test_macos_sunshine_config_preserves_explicit_loopback_device(tmp_path):
    output = _sunshine_config_output(tmp_path)
    output._sunshine_config.write_text(
        "port = 47989\naudio_sink = BlackHole 2ch\n", encoding="utf-8"
    )

    with patch("streaming.moonlight_output.sys.platform", "darwin"):
        output._ensure_service_files(48089)

    config = output._sunshine_config.read_text(encoding="utf-8")
    assert "audio_sink = BlackHole 2ch\n" in config


def test_non_macos_sunshine_config_keeps_default_audio_sink(tmp_path):
    output = _sunshine_config_output(tmp_path)

    with patch("streaming.moonlight_output.sys.platform", "win32"):
        output._ensure_service_files(48089)

    config = output._sunshine_config.read_text(encoding="utf-8")
    assert "audio_sink = default_audio_sink\n" in config


def test_sunshine_service_setup_preserves_paired_client_state(tmp_path):
    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=(0, 0, 1920, 1080, 1920, 1080),
        device="cuda:0",
    )
    output._state_dir = tmp_path
    output._sunshine_config = tmp_path / "sunshine.conf"
    output._sunshine_apps = tmp_path / "apps.json"
    output._sunshine_state = tmp_path / "sunshine_state.json"
    output._sunshine_pki_dir = tmp_path / "credentials"
    output._sunshine_private_key = output._sunshine_pki_dir / "cakey.pem"
    output._sunshine_certificate = output._sunshine_pki_dir / "cacert.pem"
    saved_state = '{"root":{"uniqueid":"stable-host-id","named_devices":[]}}\n'
    output._sunshine_state.write_text(saved_state, encoding="utf-8")

    output._ensure_service_files(48089)

    assert output._sunshine_state.read_text(encoding="utf-8") == saved_state


def test_sunshine_pairing_targets_selected_request_when_multiple_are_waiting(tmp_path):
    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=(0, 0, 1920, 1080, 1920, 1080),
        device="cuda:0",
    )
    pairings = [
        {"id": "client-one", "name": "Living room"},
        {"id": "client-two", "name": "Bedroom"},
    ]
    with patch.object(
        output,
        "_sunshine_api_request",
        side_effect=[{"pairings": pairings}, {"status": True}],
    ) as request:
        output._pair_sunshine_client("1234", "client-two")

    assert request.call_args_list[1].args[1]["pairing_id"] == "client-two"
    assert request.call_args_list[1].args[1]["name"] == "Bedroom"


def test_amd_moonlight_drains_amf_backpressure_and_caps_burst_rate(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    class _Frame:
        is_cuda = True
        shape = (1080, 1920, 3)

    class _Rgba:
        device = "cuda:0"

        @staticmethod
        def data_ptr():
            return 1234

        @staticmethod
        def stride(_index):
            return 1920 * 4

        @staticmethod
        def element_size():
            return 1

    class _Encoder:
        def __init__(self):
            self.submits = []
            self.submit_results = iter((False, True, True))
            self.packets = iter(
                (
                    b"\x00\x00\x00\x01\x65",
                    None,
                    b"\x00\x00\x00\x01\x41",
                    None,
                    None,
                    None,
                )
            )

        def submit_hip_rgba(self, *args, **kwargs):
            self.submits.append((args, kwargs))
            return next(self.submit_results)

        def read_packet(self):
            return next(self.packets)

    class _Bridge:
        connected = True

        def __init__(self):
            self.sent = []

        def send_access_unit(self, *args, **kwargs):
            self.sent.append((args, kwargs))

    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(1920, 1080),
        input_bounds=None,
        device="cuda:0",
    )
    encoder = _Encoder()
    bridge = _Bridge()
    rgba = _Rgba()
    output._bridge = bridge
    output._encoder = encoder
    output._encoder_config = MoonlightClientConfig(
        1920, 1080, 60.0, 20_000, "h264", 1920, 1080,
        0, 0, 1920, 1080, 0,
    )
    output._frame_index = 4
    output._amd_pending_frames.append(3)
    output._frame_timestamps[3] = 2_000_000
    output._force_idr.set()
    output._fit_frame = lambda frame, _config: frame
    output._to_rgba8 = lambda _frame: rgba
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda **_kwargs: SimpleNamespace(cuda_stream=99)
            )
        ),
    )
    clock = iter((100.0, 100.005, 100.017))
    monkeypatch.setattr(moonlight_output.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(moonlight_output.sys, "platform", "linux")
    runtime_result = SimpleNamespace(sbs=_Frame())

    output.submit_runtime_result(runtime_result, 12.5)
    output.submit_runtime_result(runtime_result, 12.505)
    output.submit_runtime_result(runtime_result, 12.517)

    assert len(encoder.submits) == 3  # full queue retry, then one later paced frame
    assert [call[1]["force_idr"] for call in encoder.submits] == [True, True, False]
    assert [item[0][0] for item in bridge.sent] == [3, 4]
    assert [item[0][1] for item in bridge.sent] == [2_000_000, 12_500_000_000]
    assert [item[1]["idr"] for item in bridge.sent] == [True, False]
    assert not output._force_idr.is_set()
    assert output._frame_index == 6
    assert list(output._amd_pending_frames) == [5]


def test_amd_encoder_wrapper_reports_amf_input_full_without_raising():
    from streaming.amd_encoder import AmdAmfSurfaceEncoder

    class _Library:
        result = 0

        def d2s_amd_encoder_submit_hip_rgba(self, *_args):
            return self.result

        @staticmethod
        def d2s_amd_encoder_destroy(_handle):
            pass

    encoder = AmdAmfSurfaceEncoder.__new__(AmdAmfSurfaceEncoder)
    encoder._dll = _Library()
    encoder._abi_version = 4
    encoder._handle = 1

    assert encoder.submit_hip_rgba(1234, 7680, 99) is False
    encoder._dll.result = 1
    assert encoder.submit_hip_rgba(1234, 7680, 99) is True


def test_amd_encoder_wrapper_resizes_oversized_access_unit():
    import ctypes

    from streaming.amd_encoder import AmdAmfSurfaceEncoder

    payload = b"x" * (2 * 1024 * 1024 + 1)

    class _Library:
        calls = 0

        def d2s_amd_encoder_read_packet(self, _handle, output, capacity):
            self.calls += 1
            if self.calls == 1:
                assert capacity == 2 * 1024 * 1024
                return -len(payload)
            assert capacity == len(payload)
            ctypes.memmove(output, payload, len(payload))
            return len(payload)

        @staticmethod
        def d2s_amd_encoder_destroy(_handle):
            pass

    encoder = AmdAmfSurfaceEncoder.__new__(AmdAmfSurfaceEncoder)
    encoder._dll = _Library()
    encoder._handle = 1
    encoder._packet_capacity = 2 * 1024 * 1024
    encoder._packet_buffer = ctypes.create_string_buffer(encoder._packet_capacity)

    assert encoder.read_packet() == payload
    assert encoder._dll.calls == 2


@pytest.mark.parametrize("display_mode", ["Full-SBS", "Half-TAB", "Full-TAB"])
@pytest.mark.parametrize("fit_mode", ["contain", "cover", "stretch"])
@pytest.mark.parametrize("source_size", [(3840, 2160), (2560, 1080), (720, 1280), (1366, 768)])
def test_moonlight_input_regions_follow_each_actual_eye(display_mode, fit_mode, source_size):
    from streaming.aspect import presentation_blit_regions

    width, height = source_size
    output = MoonlightSbsOutput(
        base_dir=".", fps=60, display_mode=display_mode, fit_mode=fit_mode,
        input_size=source_size, input_bounds=None, device="cuda:0",
    )
    config = MoonlightClientConfig(3840, 2160, 60.0, 20_000, "hevc", width, height, 0, 0, width, height, 0)
    response = output._client_ready_response(config)
    mappings = response["input_regions"]
    assert response["accepted_width"] == 3840 and response["accepted_height"] == 2160
    assert len(mappings) == 2
    packed = (width * 2, height) if display_mode == "Full-SBS" else (
        (width, height * 2) if display_mode == "Full-TAB" else (width, height)
    )
    expected_regions = presentation_blit_regions(packed, (3840, 2160), fit_mode, display_mode, source_size)
    centers = []
    for mapping, (_crop, viewport) in zip(mappings, expected_regions):
        assert mapping[:4] == list(viewport)
        x0, y0, x1, y1 = mapping[4:]
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
        if fit_mode != "cover":
            assert (x0, y0, x1, y1) == (0, 0, width, height)
    assert centers[0] == centers[1]


def test_moonlight_full_sbs_pointer_uses_visible_rect_not_letterbox():
    output = MoonlightSbsOutput(
        base_dir=".", fps=60, display_mode="Full-SBS", fit_mode="contain",
        input_size=(1920, 1080), input_bounds=None, device="cuda:0",
    )
    config = MoonlightClientConfig(1920, 1080, 60.0, 20_000, "hevc", 1920, 1080, 0, 0, 1920, 1080, 0)
    assert output._client_ready_response(config)["input_regions"] == [
        [0, 270, 960, 810, 0, 0, 1920, 1080],
        [960, 270, 1920, 810, 0, 0, 1920, 1080],
    ]


def test_moonlight_half_sbs_retains_legacy_input_contract():
    output = MoonlightSbsOutput(
        base_dir=".", fps=60, display_mode="Half-SBS", fit_mode="contain",
        input_size=(1920, 1080), input_bounds=None, device="cuda:0",
    )
    config = MoonlightClientConfig(1920, 1080, 60.0, 20_000, "hevc", 1920, 1080, 0, 0, 1920, 1080, 0)
    assert output._client_ready_response(config) == {"accepted_width": 1920, "accepted_height": 1080}


@pytest.mark.parametrize("display_mode", ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB"])
@pytest.mark.parametrize("fit_mode", ["contain", "cover", "stretch"])
@pytest.mark.parametrize("percent", ["100%", "85%", "75%", "50%"])
def test_moonlight_gpu_fit_preserves_existing_pixels(display_mode, fit_mode, percent):
    import torch
    from stereo_runtime.pipeline import _resolve_pipeline_render_size
    from stereo_runtime.render_size import RenderSizeConfig
    from streaming.aspect import apply_aspect_on_gpu

    if not torch.cuda.is_available():
        pytest.skip("GPU byte-parity check requires CUDA or ROCm")
    original = (256, 144)
    width, height = _resolve_pipeline_render_size(original, RenderSizeConfig(scale_factor=percent))
    source = (width * (2 if display_mode == "Full-SBS" else 1),
              height * (2 if display_mode == "Full-TAB" else 1))
    frame = torch.randint(0, 256, (source[1], source[0], 3), device="cuda", dtype=torch.uint8)
    output = MoonlightSbsOutput(
        base_dir=".", fps=60, display_mode=display_mode, fit_mode=fit_mode,
        input_size=original, input_bounds=None, device="cuda:0",
    )
    config = MoonlightClientConfig(256, 144, 60.0, 20_000, "hevc", *original, 0, 0, *original, 0)
    expected = frame if source == original else apply_aspect_on_gpu(
        frame, source, original, fit_mode, display_mode, original,
    )
    actual = output._fit_frame(frame, config)
    assert torch.equal(actual, expected)
    assert actual.is_cuda and actual.dtype == torch.uint8


@pytest.mark.parametrize("has_event", [True, False])
def test_moonlight_encoder_stream_waits_for_frame_producer(has_event):
    from types import SimpleNamespace
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU stream handoff requires CUDA or ROCm")
    output = MoonlightSbsOutput(
        base_dir=".", fps=60, display_mode="Half-SBS", fit_mode="contain",
        input_size=(64, 32), input_bounds=None, device="cuda:0",
    )
    producer = torch.cuda.Stream() if has_event else torch.cuda.current_stream()
    with torch.cuda.stream(producer):
        frame = torch.full((32, 64, 3), 173, device="cuda", dtype=torch.uint8)
        ready = torch.cuda.Event()
        ready.record()
    result = SimpleNamespace(sbs=frame, cuda_ready_event=ready if has_event else None)
    with output._encoder_frame_context(result, frame):
        assert torch.cuda.current_stream() == output._encode_stream
        assert output._encode_stream != producer
        copied = frame.clone()
    torch.cuda.synchronize()
    assert torch.equal(copied, frame)


def test_moonlight_area_compiler_failure_uses_compatibility_path(monkeypatch):
    import torch
    import streaming.aspect as aspect

    if not torch.cuda.is_available():
        pytest.skip("GPU compiler fallback check requires CUDA or ROCm")
    triton = pytest.importorskip("triton")

    def unavailable():
        raise triton.TritonError("optional compiler unsupported")

    monkeypatch.setattr(aspect, "_packed_area_blit_kernel", unavailable)
    output = MoonlightSbsOutput(
        base_dir=".", fps=60, display_mode="Full-SBS", fit_mode="contain",
        input_size=(128, 72), input_bounds=None, device="cuda:0",
    )
    frame = torch.randint(0, 256, (72, 256, 3), device="cuda", dtype=torch.uint8)
    config = MoonlightClientConfig(128, 72, 60.0, 20_000, "hevc", 128, 72, 0, 0, 128, 72, 0)
    expected = aspect.apply_aspect_on_gpu(frame, (256, 72), (128, 72), "contain", "Full-SBS", (128, 72))
    assert torch.equal(output._fit_frame(frame, config), expected)


def test_gpu_pixel_pack_retains_rounding_boundaries_and_nonfinite_values():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU packing check requires CUDA or ROCm")
    pack = pytest.importorskip("stereo_runtime.fast_plus_fused_triton").try_pack_rgb_to_uint8
    frame = torch.rand((1, 3, 108, 384), device="cuda") * 1.4 - 0.2
    ties = (torch.arange(256, device="cuda", dtype=torch.float32) + 0.5) / 255.0
    frame.reshape(-1)[:256] = ties
    frame.reshape(-1)[256:259] = torch.tensor([float("nan"), float("inf"), float("-inf")], device="cuda")
    expected = frame.clamp(0, 1).mul(255).round().to(torch.uint8)
    actual = pack(frame)
    if actual is None:
        pytest.skip("optional GPU packing runtime is unavailable")
    assert torch.equal(actual, expected)
    assert actual.shape == frame.shape and actual.device == frame.device
    assert pack(frame.transpose(-1, -2)) is None
    assert pack(frame.half()) is None


@pytest.mark.parametrize("backend,native_available", [("auto", True), ("auto", False), ("pynv", True)])
def test_cuda_moonlight_prefers_zero_delay_nv12_and_preserves_explicit_or_missing_bridge_fallback(
    monkeypatch, tmp_path, backend, native_available,
):
    import sys
    from types import SimpleNamespace
    from streaming import direct_sbs, nvidia_encoder, nvenc_cudaarray_bridge

    calls = []
    encoder = SimpleNamespace(close=lambda: None)
    device = SimpleNamespace(type="cuda", index=0)
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip=None), device=lambda _value: device,
        cuda=SimpleNamespace(is_available=lambda: True, Stream=lambda **_kwargs: SimpleNamespace(cuda_stream=789)),
    )

    def native(width, height, **kwargs):
        calls.append(("native", width, height, kwargs))
        if not native_available:
            raise RuntimeError("optional direct NV12 bridge unavailable")
        return encoder

    def legacy(codec, width, height, **kwargs):
        calls.append(("pynv", width, height, kwargs))
        assert codec == "codec"
        return encoder

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(moonlight_output.sys, "platform", "win32")
    monkeypatch.setattr(nvenc_cudaarray_bridge, "NvencCudaNv12Encoder", native)
    monkeypatch.setattr(nvidia_encoder, "PyNvVideoCodecEncoder", legacy)
    monkeypatch.setattr(direct_sbs, "_load_pynvvideo_codec", lambda: "codec")
    output = _sunshine_config_output(tmp_path)
    output.encoder_backend = backend
    config = MoonlightClientConfig(3840, 2160, 60.0, 20000, "hevc", 1920, 1080, 0, 0, 1920, 1080, 0)
    try:
        assert output._configure_encoder(config) == {"accepted_width": 3840, "accepted_height": 2160}
        assert output._encoder is encoder and output._encoder_config is config
        assert output.current_network_bitrate_mbps == 20.0
        expected = "native" if backend == "auto" and native_available else "pynv"
        assert calls[-1][0] == expected
        assert calls[-1][1:3] == (3840, 2160)
        assert calls[-1][3]["bitrate"] == 20_000_000
        assert calls[-1][3]["fps"] == 60 and calls[-1][3]["hevc"] is True
        if expected == "pynv":
            assert calls[-1][3]["cuda_stream"] == 789
        if backend == "pynv":
            assert len(calls) == 1
    finally:
        output._client_disconnected()


def test_nvenc_direct_nv12_uses_the_active_encoder_stream(monkeypatch):
    import sys
    from types import SimpleNamespace
    from streaming import nvidia_encoder
    from streaming.nvenc_cudaarray_bridge import NvencCudaNv12Encoder

    calls = []
    sequence = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_stream=lambda *, device: SimpleNamespace(
                cuda_stream=456, synchronize=lambda: sequence.append("sync")
            )
        )
    )

    def convert(frame, **kwargs):
        calls.append((frame, kwargs))
        sequence.append("convert")
        return None

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(nvidia_encoder, "rgb_cuda_to_nv12", convert)
    encoder = object.__new__(NvencCudaNv12Encoder)
    encoder._closed = False
    encoder._handle = object()
    encoder._nv12 = SimpleNamespace(device=object())
    encoder._library = SimpleNamespace(
        d2s_nvenc_nv12_submit=lambda *_args: sequence.append("submit") or 0
    )
    encoder._drain_timed_packets = lambda: []

    assert encoder.encode_access_units(object(), frame_index=7) == []
    assert calls[0][1]["encoder_stream"] == 456
    assert sequence == ["convert", "sync", "submit"]


def test_moonlight_encoder_target_uses_scaled_transport_size(tmp_path):
    output = MoonlightSbsOutput(
        base_dir=tmp_path,
        fps=60,
        display_mode="Half-SBS",
        fit_mode="contain",
        input_size=(3840, 2160),
        input_bounds=None,
        device="cuda:0",
        target_size=(3264, 1836),
    )
    client_config = MoonlightClientConfig(
        3840, 2160, 60.0, 20_000, "hevc", 3840, 2160, 0, 0, 3840, 2160, 0,
    )

    effective = output._encoder_config_for_target(client_config)
    smaller_client = output._encoder_config_for_target(
        MoonlightClientConfig(
            1920, 1080, 60.0, 20_000, "hevc", 3840, 2160, 0, 0, 3840, 2160, 0,
        )
    )

    assert (effective.width, effective.height) == (3264, 1836)
    assert (smaller_client.width, smaller_client.height) == (1920, 1080)
