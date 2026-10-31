from __future__ import annotations

import json
import socket
import struct
import threading

from streaming.moonlight_bridge import MoonlightBridgeServer, MoonlightClientConfig


def _hello(token: str) -> bytes:
    return (
        json.dumps(
            {
                "type": "hello",
                "token": token,
                "width": 3840,
                "height": 2160,
                "fps_x100": 6000,
                "bitrate_kbps": 30000,
                "codec": "h264",
                "source_width": 3840,
                "source_height": 2160,
                "source_left": 0,
                "source_top": 0,
                "desktop_width": 3840,
                "desktop_height": 2160,
                "dynamic_range": 0,
            }
        )
        + "\n"
    ).encode("utf-8")


def _read_exact(sock: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = sock.recv(size - len(output))
        if not chunk:
            raise EOFError("socket closed early")
        output.extend(chunk)
    return bytes(output)


def test_moonlight_bridge_authenticates_and_preserves_encoded_frame_metadata():
    configured: list[MoonlightClientConfig] = []
    idr_requested = threading.Event()
    bridge = MoonlightBridgeServer(
        "a" * 64,
        on_client_start=lambda config: configured.append(config) or {"accepted_width": config.width},
        on_client_stop=lambda: None,
        on_idr_request=idr_requested.set,
    )
    host, port = bridge.start()

    try:
        with socket.create_connection((host, port), timeout=3.0) as client:
            client.sendall(_hello("a" * 64))
            response = json.loads(client.makefile("rb").readline())
            assert response == {"ok": True, "accepted_width": 3840}
            assert configured == [
                MoonlightClientConfig(
                    width=3840,
                    height=2160,
                    fps=60.0,
                    bitrate_kbps=30000,
                    codec="h264",
                    source_width=3840,
                    source_height=2160,
                    source_left=0,
                    source_top=0,
                    desktop_width=3840,
                    desktop_height=2160,
                    dynamic_range=0,
                )
            ]

            assert bridge.send_access_unit(7, 123456789, b"\x00\x00\x00\x01\x65frame", idr=True)
            header = _read_exact(client, struct.calcsize("!4sQQII"))
            marker, frame_index, timestamp, flags, payload_size = struct.unpack(
                "!4sQQII", header
            )
            payload = _read_exact(client, payload_size)
            assert (marker, frame_index, timestamp, flags) == (
                b"D2SF",
                7,
                123456789,
                1,
            )
            assert payload == b"\x00\x00\x00\x01\x65frame"

            client.sendall(b"I")
            assert idr_requested.wait(1.0)
    finally:
        bridge.close()


def test_moonlight_bridge_rejects_bad_token_and_unsupported_configuration():
    bridge = MoonlightBridgeServer(
        "b" * 64,
        on_client_start=lambda _config: {"ok": True},
        on_client_stop=lambda: None,
        on_idr_request=lambda: None,
    )
    host, port = bridge.start()

    try:
        with socket.create_connection((host, port), timeout=3.0) as client:
            client.sendall(_hello("wrong"))
            assert json.loads(client.makefile("rb").readline())["error"] == "unauthorized"

        unsupported = json.loads(_hello("b" * 64))
        unsupported["codec"] = "av1"
        with socket.create_connection((host, port), timeout=3.0) as client:
            client.sendall((json.dumps(unsupported) + "\n").encode("utf-8"))
            response = json.loads(client.makefile("rb").readline())
            assert response["ok"] is False
            assert "unsupported" in response["error"]
    finally:
        bridge.close()
