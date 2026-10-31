from __future__ import annotations

import json
import ssl
from unittest.mock import patch

from streaming import moonlight_output
from streaming.moonlight_output import MoonlightSbsOutput, read_sunshine_base_port
from streaming.moonlight_bridge import MoonlightClientConfig
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
            return {"data": b"annex-b", "frame_index": 0, "idr": True}

    class _Bridge:
        connected = True

        def __init__(self):
            self.sent = []

        def send_access_unit(self, *args, **kwargs):
            self.sent.append((args, kwargs))
            return True

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

    with patch("streaming.moonlight_output.sys.platform", "darwin"):
        output.submit_runtime_result(result, timestamp)

    assert native_frame.released
    assert result.native_stream_frame is None
    args, kwargs = bridge.sent[0]
    assert args == (0, 12_500_000_000, b"annex-b")
    assert kwargs == {"idr": True}


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

        def pack_metal(self, destination, destination_size, output_size, output_format):
            self.packed = (destination, destination_size, output_size, output_format)

    class _Library:
        def d2s_videotoolbox_source_buffer(self, _handle, width, height, _error, _size):
            self.source_size = (width, height)
            return 1234

        def d2s_videotoolbox_encode(
            self, _handle, _frame_index, _force, region_count, regions,
            output, output_size, keyframe, _error, _error_size,
        ):
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
    encoder._abi_version = 3
    encoder._handle = 1

    assert encoder.submit_hip_rgba(1234, 7680, 99) is False
    encoder._dll.result = 1
    assert encoder.submit_hip_rgba(1234, 7680, 99) is True
