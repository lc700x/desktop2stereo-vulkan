"""Moonlight output using a bundled Sunshine process and GPU encoders."""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
import queue
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from streaming.aspect import normalize_display_fit_mode
from streaming.moonlight_bridge import MoonlightBridgeServer, MoonlightClientConfig
from streaming.moonlight_paths import (
    moonlight_state_directory,
    sunshine_executable,
    sunshine_working_directory,
)

_SUNSHINE_DEFAULT_PORT = 47989
_SUNSHINE_MIN_PORT = 1019
_SUNSHINE_MAX_PORT = 65514
_SUNSHINE_PORT_OFFSETS = (
    (socket.SOCK_STREAM, (-5, 0, 1, 21)),
    (socket.SOCK_DGRAM, (9, 10, 11, 13)),
)


@dataclass
class _NativeFrameSubmission:
    native_frame: Any
    output_format: str
    frame_index: int
    capture_timestamp_ns: int
    session_generation: int
    client_generation: int
    force_keyframe: bool


def read_sunshine_base_port(config_path: str | Path) -> int:
    try:
        for line in Path(config_path).read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip().casefold() == "port":
                port = int(value.split("#", 1)[0].strip())
                if _SUNSHINE_MIN_PORT <= port <= _SUNSHINE_MAX_PORT:
                    return port
    except (OSError, ValueError):
        pass
    return _SUNSHINE_DEFAULT_PORT


def read_sunshine_certificate_path(
    config_path: str | Path,
    fallback: str | Path,
) -> Path:
    config_path = Path(config_path)
    try:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip().casefold() == "cert":
                path = Path(value.split("#", 1)[0].strip().strip('"'))
                return path if path.is_absolute() else config_path.parent / path
    except OSError:
        pass
    return Path(fallback)


def _request_sunshine_api(
    *,
    state_dir: Path,
    base_port: int,
    username: str,
    password: str,
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    if not endpoint.startswith("/api/") or "://" in endpoint:
        raise ValueError("Sunshine API endpoint must be a local /api path.")
    token = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
    headers = {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    certificate = read_sunshine_certificate_path(
        state_dir / "sunshine.conf",
        state_dir / "sunshine-pki" / "cacert.pem",
    )
    expected_certificate = ssl.PEM_cert_to_DER_cert(
        certificate.read_text(encoding="ascii")
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    # The GUI runtime injects truststore globally. Its wrapper verifies the
    # macOS system trust chain even for CERT_NONE contexts, while Sunshine's
    # local self-signed certificate is intentionally validated by the exact
    # certificate fingerprint below. Use truststore's underlying stdlib
    # context when present so the explicit pin remains the sole trust check.
    context = getattr(context, "_ctx", context)
    connection = http.client.HTTPSConnection(
        "127.0.0.1",
        base_port + 1,
        context=context,
        timeout=timeout,
    )
    try:
        connection.connect()
        if connection.sock is None:
            raise ssl.SSLError("Sunshine TLS connection has no peer socket.")
        peer_certificate = connection.sock.getpeercert(binary_form=True)
        if not hmac.compare_digest(
            hashlib.sha256(peer_certificate).digest(),
            hashlib.sha256(expected_certificate).digest(),
        ):
            raise ssl.SSLError(
                "Sunshine TLS certificate does not match the D2S-managed certificate."
            )
        connection.request(method, endpoint, body=data, headers=headers)
        response = connection.getresponse()
        response_body = response.read()
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Sunshine API returned HTTP {response.status}.")
        result = json.loads(response_body.decode("utf-8"))
        if not isinstance(result, dict):
            raise TypeError("Sunshine returned an invalid API response.")
        return result
    finally:
        connection.close()


def request_sunshine_api(
    state_dir: str | Path,
    method: str,
    endpoint: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    state_dir = Path(state_dir)
    account = json.loads(
        (state_dir / "web_account.json").read_text(encoding="utf-8")
    )
    username = str(account.get("username") or "")
    password = str(account.get("password") or "")
    if not username or not password:
        raise RuntimeError("The Sunshine web account is not initialized.")
    return _request_sunshine_api(
        state_dir=state_dir,
        base_port=read_sunshine_base_port(state_dir / "sunshine.conf"),
        username=username,
        password=password,
        method=method,
        endpoint=endpoint,
        payload=payload,
        timeout=timeout,
    )


class MoonlightSbsOutput:
    """Encode D2S SBS tensors and feed access units to the bundled Sunshine host."""

    protocol = "MOONLIGHT"
    synchronous_submit = True

    def __init__(
        self,
        *,
        base_dir: str | Path,
        fps: int,
        display_mode: str,
        fit_mode: str,
        input_size: tuple[int, int] | None,
        input_bounds: tuple[int, int, int, int, int, int] | None,
        device: str,
        encoder_backend: str = "auto",
        bitrate_mbps: int = 0,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.fps = max(1, int(fps))
        self.display_mode = str(display_mode or "Half-SBS")
        self.fit_mode = str(fit_mode or "contain")
        self.input_size = input_size
        self.input_bounds = input_bounds
        self.device = str(device)
        self.encoder_backend = str(encoder_backend or "auto").casefold()
        self.bitrate_mbps = max(0, int(bitrate_mbps))
        self._frame_index = 0
        self._frame_timestamps: dict[int, int] = {}
        self._amd_pending_frames: deque[int] = deque()
        self._amd_rate_update_at: float | None = None
        self._amd_rate_tokens = 1.0
        self._amd_backpressure_logged = False
        self._force_idr = threading.Event()
        self._native_session_lock = threading.Lock()
        self._native_session_generation = 0
        self._native_session_ready = False
        self._native_frame_index = 0
        self._native_force_keyframe_pending = True
        self._native_keyframe_request_generation = 0
        self._native_frame_queue: queue.Queue[_NativeFrameSubmission] = queue.Queue(maxsize=1)
        self._native_queue_lock = threading.Lock()
        self._native_worker_stop = threading.Event()
        self._native_worker: threading.Thread | None = None
        self._native_worker_lock = threading.Lock()
        self._encoder_lifecycle_lock = threading.Lock()
        self._native_metrics_lock = threading.Lock()
        self._native_metrics_started = time.monotonic()
        self._native_metrics_encoded = 0
        self._native_metrics_sent = 0
        self._native_metrics_dropped = 0
        self._native_metrics_warp_gpu_ms = 0.0
        self._native_metrics_encode_ms = 0.0
        self._native_metrics_send_ms = 0.0
        self._negotiated_bitrate_mbps = float(self.bitrate_mbps)
        self._encoder_lock = threading.RLock()
        self._encoder = None
        self._encoder_config: MoonlightClientConfig | None = None
        self._sunshine: subprocess.Popen[str] | None = None
        self._sunshine_log_thread: threading.Thread | None = None
        self._bridge: MoonlightBridgeServer | None = None
        self._state_dir = self._state_directory()
        self._server_dir = Path(__file__).resolve().parent / "moonlight_server"
        self._sunshine_exe = sunshine_executable(self._server_dir)
        self._sunshine_config = self._state_dir / "sunshine.conf"
        self._sunshine_apps = self._state_dir / "apps.json"
        self._sunshine_credentials = self._state_dir / "web_credentials"
        self._sunshine_account_file = self._state_dir / "web_account.json"
        self._sunshine_state = self._state_dir / "sunshine_state.json"
        self._sunshine_pki_dir = self._state_dir / "sunshine-pki"
        self._sunshine_private_key = self._sunshine_pki_dir / "cakey.pem"
        self._sunshine_certificate = self._sunshine_pki_dir / "cacert.pem"
        self._sunshine_port = read_sunshine_base_port(self._sunshine_config)
        self._sunshine_username: str | None = None
        self._sunshine_password: str | None = None
        pairing_request = str(os.environ.get("D2S_MOONLIGHT_PAIRING_REQUEST_FILE", "") or "").strip()
        self._pairing_request_file = Path(pairing_request) if pairing_request else None
        self._pairing_stop = threading.Event()
        self._pairing_thread: threading.Thread | None = None

    @property
    def pairing_url(self) -> str:
        return f"https://localhost:{self._sunshine_port + 1}"

    @staticmethod
    def _state_directory() -> Path:
        return moonlight_state_directory()

    @property
    def current_network_bitrate_mbps(self) -> float:
        return self._negotiated_bitrate_mbps

    def _request_keyframe(self) -> None:
        with self._native_session_lock:
            self._native_force_keyframe_pending = True
            self._native_keyframe_request_generation += 1
            self._force_idr.set()

    def _ensure_service_files(self, port: int) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._sunshine_pki_dir.mkdir(parents=True, exist_ok=True)
        managed_values = {
            "port": str(port),
            "sunshine_name": "D2S Sunshine",
            "pkey": self._sunshine_private_key.as_posix(),
            "cert": self._sunshine_certificate.as_posix(),
            "file_state": self._sunshine_state.as_posix(),
            "credentials_file": self._sunshine_credentials.as_posix(),
            "file_apps": self._sunshine_apps.as_posix(),
            "log_path": (self._state_dir / "sunshine.log").as_posix(),
        }
        default_audio_sink = "" if sys.platform == "darwin" else "default_audio_sink"
        if self._sunshine_config.exists():
            lines = self._sunshine_config.read_text(encoding="utf-8").splitlines()
            found = set()
            updated = []
            for line in lines:
                key, separator, _value = line.partition("=")
                normalized_key = key.strip().casefold()
                if normalized_key == "log_file":
                    continue
                if (
                    sys.platform == "darwin"
                    and separator
                    and normalized_key == "audio_sink"
                    and _value.split("#", 1)[0].strip().strip('"').casefold()
                    == "default_audio_sink"
                ):
                    # On macOS, Sunshine uses an empty sink to select its
                    # native system-audio tap. The generic sentinel is instead
                    # treated as the name of a microphone and disables audio.
                    line = "audio_sink ="
                if separator and normalized_key in managed_values:
                    line = f"{normalized_key} = {managed_values[normalized_key]}"
                    found.add(normalized_key)
                updated.append(line)
            for key, value in managed_values.items():
                if key not in found:
                    updated.append(f"{key} = {value}")
            self._sunshine_config.write_text("\n".join(updated) + "\n", encoding="utf-8")
        else:
            values = {
                **managed_values,
                "upnp": "disabled",
                "system_tray": "disabled",
                "origin_web_ui_allowed": "pc",
                "hevc_mode": 2,
                "av1_mode": 1,
                "audio_sink": default_audio_sink,
            }
            self._sunshine_config.write_text(
                "\n".join(
                    f"{key} = {value}" if value != "" else f"{key} ="
                    for key, value in values.items()
                )
                + "\n",
                encoding="utf-8",
            )
        if not self._sunshine_apps.exists():
            self._sunshine_apps.write_text(
                json.dumps(
                    {
                        "env": {},
                        "apps": [
                            {
                                "name": "Desktop2Stereo",
                                "detached": True,
                                "image-path": "desktop.png",
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def _ensure_sunshine_account(self) -> None:
        username = "Desktop2Stereo"
        password = None
        try:
            account = json.loads(self._sunshine_account_file.read_text(encoding="utf-8"))
            if account.get("username") == username and isinstance(account.get("password"), str):
                password = account["password"]
        except (OSError, ValueError, AttributeError):
            pass
        if not password:
            password = secrets.token_urlsafe(32)

        result = subprocess.run(
            [
                str(self._sunshine_exe),
                str(self._sunshine_config),
                "--creds",
                username,
                password,
            ],
            cwd=str(sunshine_working_directory(self._sunshine_exe)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not initialize the Sunshine account (exit code {result.returncode})."
            )
        self._sunshine_account_file.write_text(
            json.dumps({"username": username, "password": password}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._sunshine_username = username
        self._sunshine_password = password

    def _sunshine_api_request(
        self,
        method: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 20.0,
    ) -> dict[str, Any]:
        if not self._sunshine_username or not self._sunshine_password:
            raise RuntimeError("Sunshine account has not been initialized.")
        return _request_sunshine_api(
            state_dir=self._state_dir,
            base_port=self._sunshine_port,
            username=self._sunshine_username,
            password=self._sunshine_password,
            method=method,
            endpoint="/api/pin",
            payload=payload,
            timeout=timeout,
        )

    @staticmethod
    def _valid_pairing_pin(pin: str) -> bool:
        return len(pin) == 4 and pin.isascii() and pin.isdigit()

    def _pair_sunshine_client(
        self,
        pin: str,
        pairing_id: str | None = None,
    ) -> None:
        if not self._valid_pairing_pin(pin):
            print("[Moonlight] Pairing PIN must contain four digits.", flush=True)
            return
        try:
            pairings = self._sunshine_api_request("GET").get("pairings", [])
            if not pairings:
                print("[Moonlight] No Moonlight client is waiting for pairing.", flush=True)
                return
            if pairing_id:
                pairing = next(
                    (item for item in pairings if str(item.get("id", "")) == pairing_id),
                    None,
                )
                if pairing is None:
                    print("[Moonlight] The selected pairing request has expired.", flush=True)
                    return
            elif len(pairings) == 1:
                pairing = pairings[0]
            else:
                print(
                    "[Moonlight] More than one client is waiting; pair clients one at a time.",
                    flush=True,
                )
                return
            name = str(pairing.get("name") or "Moonlight client").strip()
            while len(name.encode("utf-8")) > 128:
                name = name[:-1]
            result = self._sunshine_api_request(
                "POST",
                {"pairing_id": str(pairing["id"]), "pin": pin, "name": name},
            )
            if result.get("status") is True:
                print(f"[Moonlight] Paired Moonlight client: {name}", flush=True)
            else:
                print(
                    f"[Moonlight] Pairing failed for {name}; check the PIN and retry.",
                    flush=True,
                )
        except http.client.HTTPException as exc:
            print(
                f"[Moonlight] Sunshine pairing API protocol error: {exc}",
                flush=True,
            )
        except (KeyError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            print(
                f"[Moonlight] Could not complete Sunshine pairing: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def _watch_pairing_requests(self) -> None:
        request_file = self._pairing_request_file
        if request_file is None:
            return
        last_request_id = ""
        while not self._pairing_stop.wait(0.2):
            try:
                raw = request_file.read_text(encoding="utf-8").strip()
                if not raw:
                    continue
                request = json.loads(raw)
                if not isinstance(request, dict):
                    continue
                request_id = str(request.get("request_id", ""))
                pin = str(request.get("pin", ""))
                pairing_id = str(request.get("pairing_id", "") or "")
                if not request_id or request_id == last_request_id:
                    continue
                request_file.write_text("", encoding="utf-8")
                last_request_id = request_id
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self._pair_sunshine_client(pin, pairing_id or None)

    def _start_pairing_request_watcher(self) -> None:
        if self._pairing_request_file is None:
            return
        self._pairing_request_file.parent.mkdir(parents=True, exist_ok=True)
        self._pairing_stop.clear()
        self._pairing_thread = threading.Thread(
            target=self._watch_pairing_requests,
            name="D2SMoonlightPairing",
            daemon=True,
        )
        self._pairing_thread.start()

    @classmethod
    def _port_family_available(cls, base_port: int) -> bool:
        for socket_type, offsets in _SUNSHINE_PORT_OFFSETS:
            for offset in offsets:
                port = base_port + offset
                try:
                    with socket.socket(socket.AF_INET, socket_type) as probe:
                        probe.bind(("0.0.0.0", port))
                except OSError:
                    return False
        return True

    @classmethod
    def _select_sunshine_port(cls, preferred_port: int) -> int:
        candidates = [_SUNSHINE_DEFAULT_PORT]
        if preferred_port != _SUNSHINE_DEFAULT_PORT:
            candidates.append(preferred_port)
        candidates.extend(range(48089, _SUNSHINE_MAX_PORT + 1))
        candidates.extend(range(_SUNSHINE_MIN_PORT, 48089))
        seen: set[int] = set()
        for base_port in candidates:
            if base_port in seen or not _SUNSHINE_MIN_PORT <= base_port <= _SUNSHINE_MAX_PORT:
                continue
            seen.add(base_port)
            if cls._port_family_available(base_port):
                return base_port
        raise RuntimeError("No free Sunshine port family is available.")

    def start(self) -> None:
        if self._sunshine is not None:
            return
        if not self._sunshine_exe.is_file():
            raise RuntimeError(
                "Bundled Sunshine is unavailable. Build/install the D2S Moonlight server component."
            )
        self._sunshine_port = self._select_sunshine_port(
            read_sunshine_base_port(self._sunshine_config)
        )
        self._ensure_service_files(self._sunshine_port)
        self._ensure_sunshine_account()
        token = secrets.token_hex(32)
        self._bridge = MoonlightBridgeServer(
            token,
            on_client_start=self._configure_encoder,
            on_client_stop=self._client_disconnected,
            on_idr_request=self._request_keyframe,
        )
        host, port = self._bridge.start()
        environment = os.environ.copy()
        environment.update(
            {
                "D2S_MOONLIGHT_BRIDGE_HOST": host,
                "D2S_MOONLIGHT_BRIDGE_PORT": str(port),
                "D2S_MOONLIGHT_BRIDGE_TOKEN": token,
                "D2S_MOONLIGHT_FIT_MODE": normalize_display_fit_mode(self.fit_mode),
            }
        )
        source_bounds = self.input_bounds
        if source_bounds is None:
            width, height = self.input_size or (1920, 1080)
            source_bounds = (0, 0, width, height, width, height)
        left, top, width, height, desktop_width, desktop_height = map(
            int, source_bounds
        )
        environment.update(
            {
                "D2S_MOONLIGHT_SOURCE_LEFT": str(left),
                "D2S_MOONLIGHT_SOURCE_TOP": str(top),
                "D2S_MOONLIGHT_SOURCE_WIDTH": str(width),
                "D2S_MOONLIGHT_SOURCE_HEIGHT": str(height),
                "D2S_MOONLIGHT_DESKTOP_WIDTH": str(desktop_width),
                "D2S_MOONLIGHT_DESKTOP_HEIGHT": str(desktop_height),
            }
        )
        try:
            self._sunshine = subprocess.Popen(
                [str(self._sunshine_exe), str(self._sunshine_config)],
                cwd=str(sunshine_working_directory(self._sunshine_exe)),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._bridge.close()
            self._bridge = None
            raise
        self._sunshine_log_thread = threading.Thread(
            target=self._read_sunshine_log,
            name="D2SSunshineLog",
            daemon=True,
        )
        self._sunshine_log_thread.start()
        self._start_pairing_request_watcher()
        print(
            f"[Moonlight] Sunshine host started; pairing UI: {self.pairing_url}",
            flush=True,
        )

    def _read_sunshine_log(self) -> None:
        process = self._sunshine
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                print(f"[Moonlight/Sunshine] {line.rstrip()}", flush=True)
        except (OSError, ValueError):
            return

    def _configure_encoder(self, config: MoonlightClientConfig) -> dict[str, Any] | None:
        if config.width > 8192 or config.height > 8192 or config.fps > 120:
            return None
        try:
            if sys.platform == "darwin":
                if self.encoder_backend not in {"auto", "videotoolbox"}:
                    raise RuntimeError(
                        "Moonlight on macOS requires the Apple VideoToolbox encoder"
                    )
                from streaming.videotoolbox_encoder import VideoToolboxEncoder

                with self._native_session_lock:
                    self._native_session_generation += 1
                    session_generation = self._native_session_generation
                    self._native_session_ready = False
                    self._native_frame_index = 0
                    self._native_force_keyframe_pending = True
                    self._native_keyframe_request_generation += 1
                    self._force_idr.set()
                    self._negotiated_bitrate_mbps = config.bitrate_kbps / 1000.0
                self._drop_pending_native_frames()
                with self._encoder_lifecycle_lock:
                    with self._native_session_lock:
                        if session_generation != self._native_session_generation:
                            return None
                    with self._encoder_lock:
                        self._close_encoder_locked()
                        self._encoder = VideoToolboxEncoder(
                            config.width,
                            config.height,
                            round(config.fps),
                            config.bitrate_kbps * 1000,
                            hevc=config.codec == "hevc",
                            fit_mode=self.fit_mode,
                            display_mode=self.display_mode,
                            input_size=self.input_size,
                        )
                        self._encoder_config = config
                        self._frame_index = 0
                        self._frame_timestamps.clear()
                        self._amd_pending_frames.clear()
                        self._amd_rate_update_at = None
                        self._amd_rate_tokens = 1.0
                        self._amd_backpressure_logged = False
                    with self._native_session_lock:
                        if session_generation == self._native_session_generation:
                            self._native_session_ready = True
                        else:
                            self._native_session_ready = False
                        session_ready = self._native_session_ready
                    if not session_ready:
                        with self._encoder_lock:
                            self._close_encoder_locked()
                            self._encoder_config = None
                        return None
                print(
                    "[Moonlight] Client negotiated "
                    f"{config.width}x{config.height}@{config.fps:g}, "
                    f"{config.codec.upper()}, {config.bitrate_kbps / 1000.0:g} Mbps",
                    flush=True,
                )
                return {"accepted_width": config.width, "accepted_height": config.height}

            import torch

            if not torch.cuda.is_available() or torch.device(self.device).type != "cuda":
                return None
            use_rocm = torch.version.hip is not None
            supported_backends = {"auto", "amd"} if use_rocm else {"auto", "pynv"}
            if self.encoder_backend not in supported_backends:
                raise RuntimeError(
                    f"Moonlight requires the native {'AMF' if use_rocm else 'NVENC'} backend; "
                    f"selected backend is {self.encoder_backend}"
                )
            with self._encoder_lock:
                self._close_encoder_locked()
                if torch.version.hip is not None:
                    from streaming.amd_encoder import AmdAmfSurfaceEncoder

                    self._encoder = AmdAmfSurfaceEncoder(
                        config.width,
                        config.height,
                        round(config.fps),
                        config.bitrate_kbps * 1000,
                        hevc=config.codec == "hevc",
                        device_index=(
                            0
                            if torch.device(self.device).index is None
                            else int(torch.device(self.device).index)
                        ),
                        library_path=os.environ.get("D2S_AMD_ENCODER_DLL")
                        or self._server_dir / "d2s_amd_encoder.dll",
                    )
                    if int(getattr(self._encoder, "_abi_version", 0)) < 3:
                        self._close_encoder_locked()
                        raise RuntimeError(
                            "The bundled AMD AMF bridge lacks pooled-surface backpressure support; "
                            "rebuild d2s_amd_encoder.dll"
                        )
                else:
                    from streaming.direct_sbs import _load_pynvvideo_codec
                    from streaming.nvidia_encoder import PyNvVideoCodecEncoder

                    codec = _load_pynvvideo_codec()
                    if codec is None:
                        return None
                    device = torch.device(self.device)
                    gpu_id = 0 if device.index is None else max(0, int(device.index))
                    self._encoder = PyNvVideoCodecEncoder(
                        codec,
                        config.width,
                        config.height,
                        hevc=config.codec == "hevc",
                        fps=round(config.fps),
                        bitrate=config.bitrate_kbps * 1000,
                        gpu_id=gpu_id,
                    )
                self._encoder_config = config
                self._frame_index = 0
                self._frame_timestamps.clear()
                self._amd_pending_frames.clear()
                self._amd_rate_update_at = None
                self._amd_rate_tokens = 1.0
                self._amd_backpressure_logged = False
                self._force_idr.set()
            return {"accepted_width": config.width, "accepted_height": config.height}
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"[Moonlight] Could not configure GPU encoder: {type(exc).__name__}: {exc}", flush=True)
            self._client_disconnected()
            return None

    def _client_disconnected(self) -> None:
        with self._native_session_lock:
            self._native_session_generation += 1
            self._native_session_ready = False
            self._native_frame_index = 0
            self._native_force_keyframe_pending = False
            self._native_keyframe_request_generation += 1
            self._force_idr.clear()
            self._negotiated_bitrate_mbps = float(self.bitrate_mbps)
        self._drop_pending_native_frames()
        with self._encoder_lifecycle_lock, self._encoder_lock:
            self._close_encoder_locked()
            self._encoder_config = None
            self._frame_timestamps.clear()
            self._amd_pending_frames.clear()
            self._amd_rate_update_at = None
            self._amd_rate_tokens = 1.0
            self._amd_backpressure_logged = False

    def _ensure_native_encoder_worker(self) -> None:
        with self._native_worker_lock:
            if self._native_worker_stop.is_set():
                return
            if self._native_worker is not None and self._native_worker.is_alive():
                return
            self._native_worker = threading.Thread(
                target=self._run_native_encoder_worker,
                name="D2SMoonlightVideoToolbox",
                daemon=True,
            )
            self._native_worker.start()

    @staticmethod
    def _release_native_submission(submission: _NativeFrameSubmission) -> None:
        native_frame = submission.native_frame
        submission.native_frame = None
        release = getattr(native_frame, "release", None)
        if callable(release):
            try:
                release()
            except Exception as exc:  # noqa: BLE001 - cleanup must not strand the worker
                print(
                    f"[Moonlight] Failed to release a native stream frame: {exc}",
                    flush=True,
                )

    def _record_native_drop(self) -> None:
        with self._native_metrics_lock:
            self._native_metrics_dropped += 1

    def _drop_pending_native_frames(self) -> None:
        while True:
            with self._native_queue_lock:
                try:
                    submission = self._native_frame_queue.get_nowait()
                except queue.Empty:
                    return
            try:
                self._release_native_submission(submission)
                self._record_native_drop()
            finally:
                self._native_frame_queue.task_done()

    def _enqueue_native_frame(self, submission: _NativeFrameSubmission) -> bool:
        while not self._native_worker_stop.is_set():
            stale = None
            with self._native_queue_lock:
                if self._native_worker_stop.is_set():
                    return False
                try:
                    self._native_frame_queue.put_nowait(submission)
                    return True
                except queue.Full:
                    try:
                        stale = self._native_frame_queue.get_nowait()
                    except queue.Empty:
                        continue
            if stale is not None:
                try:
                    if stale.force_keyframe:
                        submission.force_keyframe = True
                    self._release_native_submission(stale)
                    self._record_native_drop()
                finally:
                    self._native_frame_queue.task_done()
        return False

    def _native_session_is_current(self, generation: int) -> bool:
        with self._native_session_lock:
            return (
                self._native_session_ready
                and self._native_session_generation == generation
            )

    def _run_native_encoder_worker(self) -> None:
        while not self._native_worker_stop.is_set():
            try:
                submission = self._native_frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            encode_ms = 0.0
            warp_gpu_ms = 0.0
            send_ms = 0.0
            encoded = False
            sent = False
            try:
                if not self._native_session_is_current(submission.session_generation):
                    self._record_native_drop()
                    continue
                encode_started = time.perf_counter()
                try:
                    with self._encoder_lock:
                        with self._native_session_lock:
                            if (
                                not self._native_session_ready
                                or self._native_session_generation
                                != submission.session_generation
                            ):
                                self._record_native_drop()
                                continue
                        encoder = self._encoder
                        config = self._encoder_config
                        if encoder is None or config is None:
                            self._record_native_drop()
                            continue
                        access_unit = encoder.encode_native_frame(
                            submission.native_frame,
                            output_format=submission.output_format,
                            frame_index=submission.frame_index,
                            force_keyframe=submission.force_keyframe,
                        )
                        pack_metrics = getattr(
                            submission.native_frame, "last_pack_metrics", {}
                        )
                        if isinstance(pack_metrics, dict):
                            try:
                                warp_gpu_ms = max(
                                    0.0, float(pack_metrics.get("metal_gpu_ms", 0.0))
                                )
                            except (TypeError, ValueError):
                                warp_gpu_ms = 0.0
                finally:
                    encode_ms = (time.perf_counter() - encode_started) * 1000.0
                    # The native Metal surface is no longer needed once the
                    # VideoToolbox call has copied and submitted it. Release it
                    # before any potentially backpressured network write.
                    self._release_native_submission(submission)
                encoded = True
                if not self._native_session_is_current(submission.session_generation):
                    self._record_native_drop()
                    continue
                bridge = self._bridge
                if bridge is None or not bridge.connected:
                    self._record_native_drop()
                    continue
                payload = bytes(access_unit["data"])
                send_started = time.perf_counter()
                sent = bridge.send_access_unit(
                    submission.frame_index,
                    submission.capture_timestamp_ns,
                    payload,
                    idr=bool(access_unit.get("idr"))
                    or self._is_idr(payload, config.codec),
                    client_generation=submission.client_generation,
                )
                send_ms = (time.perf_counter() - send_started) * 1000.0
                if not sent:
                    self._record_native_drop()
            except Exception as exc:  # noqa: BLE001 - retire a failed client, keep the worker alive
                self._record_native_drop()
                print(
                    f"[Moonlight] VideoToolbox stream worker failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                bridge = self._bridge
                disconnect = getattr(bridge, "disconnect_client", None)
                if callable(disconnect):
                    try:
                        disconnect(submission.client_generation)
                    except Exception as disconnect_error:  # noqa: BLE001 - teardown is best-effort
                        print(
                            f"[Moonlight] Could not disconnect failed client: {disconnect_error}",
                            flush=True,
                        )
            finally:
                self._release_native_submission(submission)
                self._native_frame_queue.task_done()
                self._record_native_worker_result(
                    encode_ms=encode_ms,
                    warp_gpu_ms=warp_gpu_ms,
                    send_ms=send_ms,
                    encoded=encoded,
                    sent=sent,
                )

    def _record_native_worker_result(
        self,
        *,
        encode_ms: float,
        warp_gpu_ms: float,
        send_ms: float,
        encoded: bool,
        sent: bool,
    ) -> None:
        now = time.monotonic()
        with self._native_metrics_lock:
            self._native_metrics_encoded += int(encoded)
            self._native_metrics_sent += int(sent)
            self._native_metrics_warp_gpu_ms += warp_gpu_ms
            self._native_metrics_encode_ms += encode_ms
            self._native_metrics_send_ms += send_ms
            elapsed = now - self._native_metrics_started
            if elapsed < 5.0:
                return
            encoded_count = self._native_metrics_encoded
            sent_count = self._native_metrics_sent
            dropped_count = self._native_metrics_dropped
            warp_gpu_total = self._native_metrics_warp_gpu_ms
            encode_total = self._native_metrics_encode_ms
            send_total = self._native_metrics_send_ms
            self._native_metrics_started = now
            self._native_metrics_encoded = 0
            self._native_metrics_sent = 0
            self._native_metrics_dropped = 0
            self._native_metrics_warp_gpu_ms = 0.0
            self._native_metrics_encode_ms = 0.0
            self._native_metrics_send_ms = 0.0
        print(
            "[Moonlight] VideoToolbox: "
            f"encoded={encoded_count / elapsed:.1f} fps, "
            f"sent={sent_count / elapsed:.1f} fps, "
            f"dropped={dropped_count}, "
            f"stereo_gpu={warp_gpu_total / max(1, encoded_count):.1f} ms, "
            f"encode={encode_total / max(1, encoded_count):.1f} ms, "
            f"send={send_total / max(1, sent_count):.1f} ms",
            flush=True,
        )

    def _stop_native_encoder_worker(self) -> None:
        self._native_worker_stop.set()
        self._drop_pending_native_frames()
        with self._native_worker_lock:
            worker = self._native_worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
            if worker.is_alive():
                print(
                    "[Moonlight] VideoToolbox worker did not stop within 3 seconds; "
                    "waiting for the active encoder operation before releasing it",
                    flush=True,
                )

    def _close_encoder_locked(self) -> None:
        encoder = self._encoder
        self._encoder = None
        if encoder is not None:
            close = getattr(encoder, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _to_rgba8(frame: Any) -> Any:
        import torch

        image = frame.detach()
        if image.ndim == 4:
            if int(image.shape[0]) != 1:
                raise ValueError("Moonlight encoder accepts one SBS frame at a time")
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"unsupported Moonlight frame shape: {tuple(image.shape)}")
        if int(image.shape[0]) in (3, 4):
            image = image[:3].permute(1, 2, 0)
        if int(image.shape[-1]) != 3:
            raise ValueError("Moonlight encoder requires a three-channel SBS frame")
        if image.dtype != torch.uint8:
            image = image.clamp(0, 1).mul(255).round().to(torch.uint8)
        alpha = torch.full(
            (*image.shape[:2], 1), 255, dtype=torch.uint8, device=image.device
        )
        return torch.cat((image, alpha), dim=-1).contiguous()

    def _fit_frame(self, frame: Any, config: MoonlightClientConfig) -> Any:
        height, width = int(frame.shape[-2]), int(frame.shape[-1])
        if int(frame.shape[0]) not in (1, 3, 4):
            height, width = int(frame.shape[0]), int(frame.shape[1])
        if (width, height) == (config.width, config.height):
            return frame
        from streaming.aspect import apply_aspect_on_gpu

        return apply_aspect_on_gpu(
            frame,
            source_size=(width, height),
            target_size=(config.width, config.height),
            fit_mode=self.fit_mode,
            display_mode=self.display_mode,
            input_size=self.input_size,
        )

    @staticmethod
    def _is_idr(payload: bytes, codec: str) -> bool:
        nals = []
        index = 0
        while index + 4 < len(payload):
            if payload[index:index + 4] == b"\x00\x00\x00\x01":
                start = index + 4
            elif payload[index:index + 3] == b"\x00\x00\x01":
                start = index + 3
            else:
                index += 1
                continue
            if start < len(payload):
                nals.append(payload[start])
            index = start + 1
        if codec == "h264":
            return any((nal & 0x1F) == 5 for nal in nals)
        return any(((nal >> 1) & 0x3F) in {19, 20, 21} for nal in nals)

    def _submit_amd_rgba(
        self,
        encoder: Any,
        rgba: Any,
        capture_ns: int,
        *,
        force_idr: bool,
    ) -> bool:
        import torch

        frame_index = self._frame_index
        accepted = encoder.submit_hip_rgba(
            int(rgba.data_ptr()),
            int(rgba.stride(0) * rgba.element_size()),
            int(torch.cuda.current_stream(device=rgba.device).cuda_stream),
            force_idr=force_idr,
        )
        if accepted is False:
            return False
        self._frame_index += 1
        self._frame_timestamps[frame_index] = capture_ns
        self._amd_pending_frames.append(frame_index)
        if len(self._frame_timestamps) > 64:
            self._frame_timestamps.pop(min(self._frame_timestamps))
        if force_idr:
            self._force_idr.clear()
        return True

    def _drain_amd_packets(
        self,
        encoder: Any,
        config: MoonlightClientConfig,
        capture_ns: int,
    ) -> list[dict[str, Any]]:
        access_units: list[dict[str, Any]] = []
        for _ in range(32):
            payload = encoder.read_packet()
            if not payload:
                break
            if not self._amd_pending_frames:
                continue
            source_index = self._amd_pending_frames.popleft()
            access_units.append(
                {
                    "data": payload,
                    "frame_index": source_index,
                    "idr": self._is_idr(payload, config.codec),
                }
            )
            self._frame_timestamps.setdefault(source_index, capture_ns)
        return access_units

    def submit_runtime_result(self, runtime_result: Any, capture_timestamp: float) -> None:
        if sys.platform == "darwin":
            native_frame = getattr(runtime_result, "native_stream_frame", None)
            ownership_transferred = False
            try:
                bridge = self._bridge
                if bridge is None or not bridge.connected:
                    return
                if native_frame is None:
                    raise RuntimeError(
                        "GPU-resident Moonlight streaming on macOS needs the native "
                        "CoreML/Metal stream path. Enable CoreML and grant Screen Recording permission."
                    )
                with self._native_session_lock:
                    if not self._native_session_ready:
                        return
                    session_generation = self._native_session_generation
                    frame_index = self._native_frame_index
                    self._native_frame_index += 1
                    keyframe_request_generation = (
                        self._native_keyframe_request_generation
                    )
                    force_keyframe = (
                        self._native_force_keyframe_pending or self._force_idr.is_set()
                    )
                submission = _NativeFrameSubmission(
                    native_frame=native_frame,
                    output_format=str(
                        getattr(runtime_result, "output_format", "half_sbs")
                        or "half_sbs"
                    ),
                    frame_index=frame_index,
                    capture_timestamp_ns=max(
                        0, int(float(capture_timestamp) * 1_000_000_000)
                    ),
                    session_generation=session_generation,
                    client_generation=int(
                        getattr(bridge, "connection_generation", 0)
                    ),
                    force_keyframe=force_keyframe,
                )
                self._ensure_native_encoder_worker()
                if not self._enqueue_native_frame(submission):
                    return
                ownership_transferred = True
                with self._native_session_lock:
                    if (
                        self._native_session_generation == session_generation
                        and self._native_keyframe_request_generation
                        == keyframe_request_generation
                    ):
                        self._native_force_keyframe_pending = False
                        self._force_idr.clear()
                try:
                    object.__setattr__(runtime_result, "native_stream_frame", None)
                except (AttributeError, TypeError):
                    pass
            finally:
                if native_frame is not None and not ownership_transferred:
                    native_frame.release()
                    try:
                        object.__setattr__(runtime_result, "native_stream_frame", None)
                    except (AttributeError, TypeError):
                        pass
            return

        frame = getattr(runtime_result, "sbs", None)
        if frame is None or not bool(getattr(frame, "is_cuda", False)):
            raise RuntimeError("Moonlight streaming requires a GPU-resident D2S SBS frame")
        bridge = self._bridge
        if bridge is None or not bridge.connected:
            return
        with self._encoder_lock:
            encoder = self._encoder
            config = self._encoder_config
            if encoder is None or config is None:
                return
            frame_index = self._frame_index
            capture_ns = max(0, int(float(capture_timestamp) * 1_000_000_000))

            if hasattr(encoder, "encode_access_units"):
                frame_index = self._frame_index
                self._frame_index += 1
                force_idr = self._force_idr.is_set()
                self._force_idr.clear()
                self._frame_timestamps[frame_index] = capture_ns
                if len(self._frame_timestamps) > 64:
                    self._frame_timestamps.pop(min(self._frame_timestamps))
                fitted_frame = self._fit_frame(frame, config)
                access_units = encoder.encode_access_units(
                    fitted_frame,
                    frame_index=frame_index,
                    force_idr=force_idr,
                )
            else:
                now = time.perf_counter()
                target_fps = max(1.0, float(config.fps))
                if self._amd_rate_update_at is None:
                    self._amd_rate_update_at = now
                else:
                    elapsed = max(0.0, now - self._amd_rate_update_at)
                    self._amd_rate_update_at = now
                    self._amd_rate_tokens = min(
                        2.0,
                        self._amd_rate_tokens + elapsed * target_fps,
                    )
                submit_due = self._amd_rate_tokens >= 0.98
                access_units = []
                if submit_due:
                    self._amd_rate_tokens -= 1.0
                    rgba = self._to_rgba8(self._fit_frame(frame, config))
                    force_idr = self._force_idr.is_set()
                    accepted = self._submit_amd_rgba(
                        encoder, rgba, capture_ns, force_idr=force_idr
                    )
                    if not accepted:
                        if not self._amd_backpressure_logged:
                            print(
                                "[Moonlight] AMD AMF input queue is full; "
                                "draining output and dropping only the current frame",
                                flush=True,
                            )
                            self._amd_backpressure_logged = True
                        access_units.extend(
                            self._drain_amd_packets(encoder, config, capture_ns)
                        )
                        accepted = self._submit_amd_rgba(
                            encoder, rgba, capture_ns, force_idr=force_idr
                        )
                    if accepted:
                        self._amd_backpressure_logged = False
                access_units.extend(
                    self._drain_amd_packets(encoder, config, capture_ns)
                )

            for access_unit in access_units:
                source_index = int(access_unit.get("frame_index", frame_index))
                timestamp_ns = self._frame_timestamps.pop(source_index, capture_ns)
                payload = bytes(access_unit["data"])
                idr = bool(access_unit.get("idr")) or self._is_idr(
                    payload, config.codec
                )
                bridge.send_access_unit(
                    source_index,
                    timestamp_ns,
                    payload,
                    idr=idr,
                )

    def submit_cuda_frame(self, frame: Any) -> None:
        class _RuntimeResult:
            sbs = frame

        self.submit_runtime_result(_RuntimeResult(), time.perf_counter())

    def submit_frame(self, frame: Any) -> None:
        raise RuntimeError("Moonlight streaming does not accept CPU video frames")

    def close(self) -> None:
        self._pairing_stop.set()
        if self._pairing_thread is not None:
            self._pairing_thread.join(timeout=1.0)
            self._pairing_thread = None
        if self._pairing_request_file is not None:
            try:
                self._pairing_request_file.write_text("", encoding="utf-8")
            except OSError:
                pass
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        self._stop_native_encoder_worker()
        self._client_disconnected()
        process = self._sunshine
        self._sunshine = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)

