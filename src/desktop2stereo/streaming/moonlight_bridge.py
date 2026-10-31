"""Authenticated local transport between D2S and the Sunshine host process."""

from __future__ import annotations

import hmac
import json
import socket
import socketserver
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_FRAME_HEADER = struct.Struct("!4sQQII")
_MAX_CONTROL_LINE = 4096
_MAX_ACCESS_UNIT = 32 * 1024 * 1024
_IDR_COMMAND = b"I"


@dataclass(frozen=True)
class MoonlightClientConfig:
    width: int
    height: int
    fps: float
    bitrate_kbps: int
    codec: str
    source_width: int
    source_height: int
    source_left: int
    source_top: int
    desktop_width: int
    desktop_height: int
    dynamic_range: int

    @classmethod
    def from_handshake(cls, value: dict[str, Any]) -> MoonlightClientConfig:
        if value.get("type") != "hello":
            raise ValueError("unexpected Sunshine bridge message")
        config = cls(
            width=int(value["width"]),
            height=int(value["height"]),
            fps=float(value["fps_x100"]) / 100.0,
            bitrate_kbps=int(value["bitrate_kbps"]),
            codec=str(value["codec"]).casefold(),
            source_width=int(value["source_width"]),
            source_height=int(value["source_height"]),
            source_left=int(value.get("source_left", 0)),
            source_top=int(value.get("source_top", 0)),
            desktop_width=int(value.get("desktop_width", value["source_width"])),
            desktop_height=int(value.get("desktop_height", value["source_height"])),
            dynamic_range=int(value.get("dynamic_range", 0)),
        )
        if (
            min(config.width, config.height, config.source_width, config.source_height) < 1
            or config.width > 16384
            or config.height > 16384
            or not 1.0 <= config.fps <= 240.0
            or not 64 <= config.bitrate_kbps <= 500_000
            or config.codec not in {"h264", "hevc"}
            or config.width % 2
            or config.height % 2
            or config.desktop_width < 1
            or config.desktop_height < 1
            or config.dynamic_range != 0
        ):
            raise ValueError("unsupported Moonlight video configuration")
        return config


class MoonlightBridgeServer:
    """Accept one Sunshine video session on loopback and send encoded access units."""

    def __init__(
        self,
        token: str,
        *,
        on_client_start: Callable[[MoonlightClientConfig], dict[str, Any] | None],
        on_client_stop: Callable[[], None],
        on_idr_request: Callable[[], None],
    ) -> None:
        if len(token) < 32:
            raise ValueError("Moonlight bridge token must be at least 32 characters")
        self.token = token
        self.on_client_start = on_client_start
        self.on_client_stop = on_client_stop
        self.on_idr_request = on_idr_request
        self._socket_lock = threading.Lock()
        self._client_socket: socket.socket | None = None
        self._client_config: MoonlightClientConfig | None = None
        self._client_reserved = False
        self._server: socketserver.ThreadingTCPServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("Moonlight bridge is not listening")
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def connected(self) -> bool:
        with self._socket_lock:
            return self._client_socket is not None

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            return self.address
        owner = self

        class _Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                line = self.rfile.readline(_MAX_CONTROL_LINE + 1)
                if len(line) > _MAX_CONTROL_LINE or not line.endswith(b"\n"):
                    return
                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise TypeError("bridge handshake must be a JSON object")
                    supplied_token = str(message.pop("token", ""))
                    if not hmac.compare_digest(supplied_token, owner.token):
                        self.wfile.write(b'{"ok":false,"error":"unauthorized"}\n')
                        return
                    config = MoonlightClientConfig.from_handshake(message)
                    with owner._socket_lock:
                        if owner._client_socket is not None or owner._client_reserved:
                            self.wfile.write(b'{"ok":false,"error":"stream already active"}\n')
                            return
                        owner._client_reserved = True
                    response = owner.on_client_start(config)
                    with owner._socket_lock:
                        owner._client_reserved = False
                        if response is None:
                            self.wfile.write(b'{"ok":false,"error":"encoder configuration rejected"}\n')
                            return
                        owner._client_socket = self.connection
                        owner._client_config = config
                    self.wfile.write((json.dumps({"ok": True, **response}) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    while True:
                        commands = self.connection.recv(256)
                        if not commands:
                            break
                        if _IDR_COMMAND in commands:
                            owner.on_idr_request()
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    try:
                        self.wfile.write(
                            (json.dumps({"ok": False, "error": str(exc)[:256]}) + "\n").encode("utf-8")
                        )
                        self.wfile.flush()
                    except OSError:
                        pass
                    return
                finally:
                    with owner._socket_lock:
                        if owner._client_socket is self.connection:
                            owner._client_socket = None
                            owner._client_config = None
                            disconnected = True
                        else:
                            disconnected = False
                        owner._client_reserved = False
                    if disconnected:
                        owner.on_client_stop()

        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="D2SMoonlightBridge",
            daemon=True,
        )
        self._server_thread.start()
        return self.address

    def send_access_unit(
        self,
        frame_index: int,
        capture_timestamp_ns: int,
        payload: bytes,
        *,
        idr: bool,
    ) -> bool:
        data = bytes(payload)
        if not data or len(data) > _MAX_ACCESS_UNIT:
            raise ValueError("encoded Moonlight access unit has invalid size")
        header = _FRAME_HEADER.pack(
            b"D2SF",
            int(frame_index) & 0xFFFFFFFFFFFFFFFF,
            int(capture_timestamp_ns) & 0xFFFFFFFFFFFFFFFF,
            1 if idr else 0,
            len(data),
        )
        with self._socket_lock:
            client = self._client_socket
            if client is None:
                return False
            client.sendall(header + data)
        return True

    def close(self) -> None:
        server = self._server
        if server is None:
            return
        self._server = None
        with self._socket_lock:
            client = self._client_socket
            self._client_socket = None
            self._client_config = None
            self._client_reserved = False
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
        server.shutdown()
        server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
