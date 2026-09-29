from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_MIN_PROBE_LOAD_RATIO = 0.85
_SAFE_TARGET_RATIO = 0.80
_SAFE_PEAK_RATIO = 0.90
_BITRATE_SEARCH_RESOLUTION_MBPS = 4


@dataclass(frozen=True)
class CalibrationTier:
    fps: int
    target_mbps: int
    peak_mbps: int


_CALIBRATION_RESULT_KEYS = {
    "Target FPS",
    "Stream Target FPS",
    "Use Stream Calibration",
    "Stream Target Bitrate Mbps",
    "Stream Peak Bitrate Mbps",
    "CRF",
}
_CALIBRATION_UI_KEYS = {
    "Language",
    "Show Log Panel",
    # Displaying runtime FPS only changes the GUI overlay, not the stream.
    "Show FPS",
    "Stream Key",
    "Stereo Mix",
    "Audio Delay",
    # Monitor enumeration indices are not stable across restart.  The
    # effective capture resolution tier is fingerprinted separately.
    "Monitor Index",
    # The local preview/output monitor does not change the encoded network
    # stream.  It can be re-enumerated with a different index after restart.
    "Stereo Output",
}


def build_calibration_fingerprint(settings: dict[str, Any]) -> dict[str, str]:
    fingerprint = {}
    for key in sorted(settings):
        if key in _CALIBRATION_RESULT_KEYS or key in _CALIBRATION_UI_KEYS:
            continue
        value = settings.get(key)
        if isinstance(value, (dict, list, tuple)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fingerprint[key] = "" if value is None else str(value)
    return fingerprint


def calibration_fingerprint_matches(
    saved_fingerprint: Any,
    settings: dict[str, Any],
) -> bool:
    """Compare profiles while accepting legacy fingerprints containing CRF."""
    if not isinstance(saved_fingerprint, dict):
        return False
    current = build_calibration_fingerprint(settings)
    if saved_fingerprint == current:
        return True
    # Older profiles included derived/UI-only fields.  CRF is now derived
    # from the calibrated safe bitrate, while monitor/output controls can be
    # re-enumerated after restart. Remove every field that the current
    # fingerprint builder intentionally excludes, not just the two fields
    # handled by the original compatibility code.
    legacy = {
        key: value
        for key, value in saved_fingerprint.items()
        if key not in _CALIBRATION_RESULT_KEYS and key not in _CALIBRATION_UI_KEYS
    }
    return legacy == current


def recommended_crf_for_bitrate(target_mbps: int | float) -> int:
    """Choose the automatic constant-quality value for a safe bitrate."""
    bitrate = float(target_mbps or 0.0)
    if bitrate >= 30.0:
        return 20
    if bitrate >= 25.0:
        return 23
    if bitrate >= 21.0:
        return 26
    if bitrate >= 19.0:
        return 28
    return 30


def minimum_bitrate_for_resolution(input_width: int, input_height: int) -> int:
    """Return a conservative integer floor for a stable stream profile."""
    pixels = max(1, int(input_width)) * max(1, int(input_height))
    if pixels <= 600_000:  # 480p and below
        return 2
    if pixels <= 921_600:  # 720p
        return 3
    if pixels <= 2_073_600:  # 1080p
        return 6
    if pixels <= 3_686_400:  # 1440p / 2K
        return 10
    return 20  # 4K and above


def calibration_tiers(
    maximum_fps: int,
    *,
    hevc: bool = False,
    input_width: int = 3840,
    input_height: int = 2160,
) -> list[CalibrationTier]:
    # The headset browser is fixed at 30 FPS. Calibration must measure network
    # headroom by changing bitrate, not by probing unsupported frame rates.
    del maximum_fps
    fps = 30
    bits_per_pixel = 0.075 if hevc else 0.12
    target = max(
        2,
        round(
            max(1, int(input_width))
            * max(1, int(input_height))
            * fps
            * bits_per_pixel
            / 1_000_000
        ),
    )
    target = min(80 if not hevc else 100, target)
    return [CalibrationTier(fps=fps, target_mbps=target, peak_mbps=round(target * 1.15))]


def evaluate_calibration_window(
    tier: CalibrationTier,
    receiver_reports: list[dict[str, Any]],
    sender_report: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    decoded = [float(item.get("decoded_fps", 0.0) or 0.0) for item in receiver_reports]
    dropped = sum(max(0, int(item.get("dropped_frames", 0) or 0)) for item in receiver_reports)
    freezes = sum(max(0, int(item.get("freeze_count", 0) or 0)) for item in receiver_reports)
    packet_loss = sum(max(0, int(item.get("packets_lost", 0) or 0)) for item in receiver_reports)
    packets_received = sum(max(0, int(item.get("packets_received", 0) or 0)) for item in receiver_reports)
    packet_total = packet_loss + packets_received
    packet_loss_rate = packet_loss / packet_total if packet_total else 0.0
    received_bitrates = [
        float(item.get("bitrate_mbps", 0.0) or 0.0)
        for item in receiver_reports
        if float(item.get("bitrate_mbps", 0.0) or 0.0) > 0.0
    ]
    jitter = [float(item.get("jitter_buffer_ms", 0.0) or 0.0) for item in receiver_reports]
    decoded_fps = statistics.median(decoded) if decoded else 0.0
    sender_fps = float((sender_report or {}).get("submitted_fps", 0.0) or 0.0)
    encoded_bitrates = [
        float(item.get("encoded_bitrate_mbps", 0.0) or 0.0)
        for item in (sender_report or {}).get("encoded_bitrates", [])
        if float(item.get("encoded_bitrate_mbps", 0.0) or 0.0) > 0.0
    ]
    received_bitrate = (
        statistics.median(received_bitrates) if received_bitrates else 0.0
    )
    encoded_bitrate = statistics.median(encoded_bitrates) if encoded_bitrates else 0.0
    probe_bitrate = encoded_bitrate or received_bitrate
    probe_load_ratio = probe_bitrate / max(1.0, float(tier.target_mbps))
    decoded_frames = max(1.0, sum(decoded))
    drop_ratio = dropped / decoded_frames
    passed = bool(
        len(receiver_reports) >= 5
        and probe_load_ratio >= _MIN_PROBE_LOAD_RATIO
        and sender_fps >= tier.fps * 0.95
        and decoded_fps >= tier.fps * 0.92
        and drop_ratio <= 0.01
        and freezes == 0
        and packet_loss == 0
        and (not jitter or statistics.median(jitter) <= 100.0)
    )
    failure_reasons = []
    if len(receiver_reports) < 5:
        failure_reasons.append("insufficient_samples")
    if probe_load_ratio < _MIN_PROBE_LOAD_RATIO:
        failure_reasons.append("insufficient_probe_bitrate")
    if sender_fps < tier.fps * 0.95:
        failure_reasons.append("sender_fps")
    if decoded_fps < tier.fps * 0.92:
        failure_reasons.append("decoded_fps")
    if drop_ratio > 0.01:
        failure_reasons.append("dropped_frames")
    if freezes:
        failure_reasons.append("freeze")
    if packet_loss:
        failure_reasons.append("packet_loss")
    if jitter and statistics.median(jitter) > 100.0:
        failure_reasons.append("jitter")
    return passed, {
        "sender_fps": round(sender_fps, 2),
        "encoded_bitrate_mbps": round(encoded_bitrate, 3),
        "probe_load_ratio": round(probe_load_ratio, 3),
        "decoded_fps": round(decoded_fps, 2),
        "drop_ratio": round(drop_ratio, 5),
        "freeze_count": freezes,
        "packets_lost": packet_loss,
        "packets_received": packets_received,
        "packet_loss_rate": round(packet_loss_rate, 6),
        "received_bitrate_mbps": round(
            received_bitrate, 3
        ),
        "jitter_buffer_ms": round(statistics.median(jitter), 2) if jitter else 0.0,
        "samples": len(receiver_reports),
        "failure_reasons": failure_reasons,
    }


class StreamCalibrationController:
    """Coordinate sender tiers with receiver-side WebRTC statistics."""

    def __init__(
        self,
        *,
        bind_port: int,
        stream_port: int,
        stream_key: str,
        maximum_fps: int,
        state_path: str | Path,
        profile_path: str | Path,
        hevc: bool = False,
        stage_seconds: float = 5.0,
        stability_seconds: float = 8.0,
        settle_seconds: float = 2.0,
        fingerprint: dict[str, str] | None = None,
        clock=time.monotonic,
    ) -> None:
        self.bind_port = int(bind_port)
        self.stream_port = int(stream_port)
        self.stream_key = str(stream_key)
        self.state_path = Path(state_path)
        self.profile_path = Path(profile_path)
        self.stage_seconds = max(2.0, float(stage_seconds))
        self.stability_seconds = max(2.0, float(stability_seconds))
        self.settle_seconds = max(0.0, float(settle_seconds))
        self.fingerprint = dict(fingerprint or {})
        self._clock = clock
        self._maximum_fps = int(maximum_fps)
        self._hevc = bool(hevc)
        self._input_resolution = (3840, 2160)
        self._tiers = calibration_tiers(maximum_fps, hevc=hevc)
        self._bitrate_limit = 100 if hevc else 80
        self._tier_index = 0
        self._pending_tier: CalibrationTier | None = self._tiers[0]
        self._active_tier = self._tiers[0]
        self._completed_stage_count = 0
        self._progress_total = self._estimate_stage_count()
        self._best_tier: CalibrationTier | None = None
        self._stable_tiers: dict[int, CalibrationTier] = {}
        self._search_upper_mbps: int | None = None
        self._decreasing_bitrate = False
        self._confirming_stability = False
        self._receiver_reports: list[dict[str, Any]] = []
        self._latest_receiver_report: dict[str, Any] = {}
        self._sender_report: dict[str, Any] | None = None
        self._sender_reports: list[dict[str, Any]] = []
        self._last_metrics: dict[str, Any] = {}
        self._result: dict[str, Any] | None = None
        self._stage_started: float | None = None
        self._measurement_started: float | None = None
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._status = "waiting_receiver"
        self._write_state()

    @staticmethod
    def _binary_search_steps(span: int) -> int:
        span = max(0, int(span))
        return max(0, (span - 1).bit_length()) if span else 0

    def _estimate_stage_count(self) -> int:
        """Estimate the largest number of probe windows for the progress bar."""
        initial = int(self._tiers[0].target_mbps)
        lower_bound = max(1, min(7, initial // 2))
        limit = int(self._bitrate_limit)
        estimate = 1 + self._binary_search_steps(initial - lower_bound) + 1
        previous = initial
        bracket_count = 1
        while previous < limit:
            candidate = min(previous * 2, limit)
            bracket_count += 1
            estimate = max(
                estimate,
                bracket_count
                + self._binary_search_steps(candidate - previous)
                + 1,
            )
            previous = candidate
        return max(1, estimate)

    @property
    def calibration_url(self) -> str:
        return f"http://127.0.0.1:{self.bind_port}/"

    def start(self) -> None:
        controller = self

        class Handler(BaseHTTPRequestHandler):
            def _client_label(self) -> str:
                address = self.client_address
                return str(address[0]) if address else "unknown"

            def _log_request(self, method: str, status: str) -> None:
                print(
                    f"[StreamCalibration] HTTP {method} "
                    f"client={self._client_label()} path={self.path} status={status}",
                    flush=True,
                )

            def _proxy_whep(self, method: str) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length) if length else None
                upstream_url = (
                    f"http://127.0.0.1:{controller.stream_port}{self.path}"
                )
                request = Request(
                    upstream_url,
                    data=body,
                    method=method,
                    headers={
                        "Accept": "application/sdp",
                        "Content-Type": self.headers.get(
                            "Content-Type", "application/sdp"
                        ),
                    },
                )
                try:
                    with urlopen(request, timeout=10) as response:
                        payload = response.read()
                        status = int(response.status)
                        location = response.headers.get("Location")
                except HTTPError as exc:
                    payload = exc.read()
                    status = int(exc.code)
                    location = None
                except URLError as exc:
                    payload = str(exc.reason).encode("utf-8", errors="replace")
                    status = 502
                    location = None
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    "application/sdp" if status < 400 else "text/plain; charset=utf-8",
                )
                if location:
                    parsed = urlsplit(location)
                    host = self.headers.get("Host", f"127.0.0.1:{controller.bind_port}")
                    location = f"http://{host}{parsed.path}"
                    if parsed.query:
                        location += f"?{parsed.query}"
                    self.send_header("Location", location)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                self._log_request(method, str(status))

            def do_GET(self):
                if self.path.rstrip("/") in {"", "/index.html"}:
                    body = controller._viewer_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    self._log_request("GET", "200")
                    return
                if self.path == "/state":
                    controller._send_json(self, controller.state())
                    self._log_request("GET", "200")
                    return
                self.send_error(404)
                self._log_request("GET", "404")

            def do_POST(self):
                if self.path.endswith("/whep"):
                    self._proxy_whep("POST")
                    return
                if self.path != "/stats":
                    self.send_error(404)
                    self._log_request("POST", "404")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    controller.add_receiver_report(payload)
                    controller._send_json(self, {"ok": True})
                    self._log_request("POST", "200")
                except Exception as exc:
                    controller._send_json(self, {"ok": False, "error": str(exc)}, 400)
                    self._log_request("POST", "400")

            def do_DELETE(self):
                if self.path.endswith("/whep"):
                    self._proxy_whep("DELETE")
                    return
                self.send_error(404)
                self._log_request("DELETE", "404")

            def log_message(self, _format, *_args):
                return

        class CalibrationHTTPServer(ThreadingHTTPServer):
            def process_request(self, request, client_address):
                client_ip = str(client_address[0]) if client_address else "unknown"
                print(
                    f"[StreamCalibration] TCP accepted client={client_ip} "
                    f"port={self.server_address[1]}",
                    flush=True,
                )
                return super().process_request(request, client_address)

        self._server = CalibrationHTTPServer(("0.0.0.0", self.bind_port), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="StreamCalibrationHTTP",
            daemon=True,
        )
        self._server_thread.start()
        print(
            f"[StreamCalibration] Headset test page: http://<PC-IP>:{self.bind_port}/",
            flush=True,
        )

    @staticmethod
    def _send_json(handler, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

    def take_pending_tier(self) -> CalibrationTier | None:
        with self._lock:
            tier = self._pending_tier
            self._pending_tier = None
            return tier

    def configure_input_resolution(self, width: int, height: int) -> None:
        """Set the selected capture resolution before the first probe starts."""
        resolution = (max(1, int(width)), max(1, int(height)))
        with self._lock:
            if resolution == self._input_resolution:
                return
            if self._measurement_started is not None or self._receiver_reports:
                return
            self._input_resolution = resolution
            self._tiers = calibration_tiers(
                self._maximum_fps,
                hevc=self._hevc,
                input_width=resolution[0],
                input_height=resolution[1],
            )
            self._tier_index = 0
            self._active_tier = self._tiers[0]
            self._completed_stage_count = 0
            self._progress_total = self._estimate_stage_count()
            self._pending_tier = self._active_tier
            self._best_tier = None
            self._stable_tiers.clear()
            self._search_upper_mbps = None
            self._decreasing_bitrate = False
            self._confirming_stability = False
            self._write_state_locked()
            print(
                f"[StreamCalibration] Input resolution: "
                f"{resolution[0]}x{resolution[1]}, initial target="
                f"{self._active_tier.target_mbps}M",
                flush=True,
            )

    def observe_sender(self, report: dict[str, Any]) -> None:
        with self._lock:
            self._sender_report = dict(report)
            self._sender_reports.append(dict(report))
            self._sender_reports = self._sender_reports[-60:]
            self._maybe_finish_stage_locked()

    def add_receiver_report(self, report: dict[str, Any]) -> None:
        with self._lock:
            if self._status == "complete":
                return
            self._latest_receiver_report = dict(report)
            decoded_fps = float(report.get("decoded_fps", 0.0) or 0.0)
            if decoded_fps <= 0.0:
                # A bitrate switch/reconnect can produce stats before the
                # headset has decoded and displayed a frame. Do not start
                # or advance the probe window in that state.
                self._status = "waiting_receiver"
                if self._measurement_started is not None:
                    # A display interruption invalidates the current window;
                    # the next decoded frame starts a fresh probe window.
                    self._stage_started = None
                    self._measurement_started = None
                    self._receiver_reports.clear()
                    self._sender_reports.clear()
                self._write_state_locked()
                return
            if self._measurement_started is None:
                # The first positive decoded FPS is the probe start event:
                # the headset is now displaying the new test stream.
                started = self._clock()
                self._stage_started = started
                self._measurement_started = started
                self._status = "testing"
                self._receiver_reports.clear()
                self._sender_reports.clear()
            self._receiver_reports.append(dict(report))
            self._receiver_reports = self._receiver_reports[-60:]
            self._maybe_finish_stage_locked()
            self._write_state_locked()

    def _maybe_finish_stage_locked(self) -> None:
        if self._measurement_started is None or self._status == "complete":
            return
        elapsed = self._clock() - self._measurement_started
        stage_seconds = (
            self.stability_seconds
            if self._confirming_stability
            else self.stage_seconds
        )
        if elapsed < stage_seconds:
            return
        passed, metrics = evaluate_calibration_window(
            self._active_tier,
            self._receiver_reports,
            {
                "submitted_fps": statistics.median(
                    float(item.get("submitted_fps", 0.0) or 0.0)
                    for item in self._sender_reports
                ) if self._sender_reports else float(self._active_tier.fps),
                "encoded_bitrates": self._sender_reports,
            },
        )
        self._last_metrics = dict(metrics)
        self._completed_stage_count += 1
        if not passed:
            print(
                f"[StreamCalibration] Probe failed target={self._active_tier.target_mbps}M "
                f"reasons={','.join(metrics.get('failure_reasons', [])) or 'unknown'} "
                f"samples={metrics.get('samples', 0)} "
                f"sender={metrics.get('sender_fps', 0)} "
                f"encoded={metrics.get('encoded_bitrate_mbps', 0)}M "
                f"decoded={metrics.get('decoded_fps', 0)} "
                f"received={metrics.get('received_bitrate_mbps', 0)}M "
                f"lost={metrics.get('packets_lost', 0)} "
                f"loss_rate={metrics.get('packet_loss_rate', 0)} "
                f"drop_ratio={metrics.get('drop_ratio', 0)} "
                f"freeze={metrics.get('freeze_count', 0)} "
                f"jitter={metrics.get('jitter_buffer_ms', 0)}ms",
                flush=True,
            )
        if passed:
            self._best_tier = self._active_tier
            self._stable_tiers[self._active_tier.target_mbps] = self._active_tier
            if self._decreasing_bitrate:
                if self._confirming_stability:
                    self._complete_locked(self._active_tier, metrics, "stable")
                    return
                upper = int(self._search_upper_mbps or self._bitrate_limit + 1)
                lower = self._active_tier.target_mbps
                if upper - lower <= _BITRATE_SEARCH_RESOLUTION_MBPS:
                    # Binary search converged to the configured resolution.
                    # Keep the highest
                    # stable candidate running for the longer confirmation.
                    self._confirming_stability = True
                    self._reset_stage_locked("confirming")
                    return
                next_target = (lower + upper) // 2
                print(
                    f"[StreamCalibration] Binary search stable={lower}M "
                    f"unstable={upper}M -> testing {next_target}M",
                    flush=True,
                )
            else:
                # Exponential bracketing finds the network ceiling in fewer
                # reconnects than fixed 5 Mbps increments. Test the ceiling
                # itself before switching to binary search.
                current_target = self._active_tier.target_mbps
                next_target = (
                    self._bitrate_limit + 1
                    if current_target >= self._bitrate_limit
                    else min(current_target * 2, self._bitrate_limit)
                )
        else:
            reasons = metrics.get("failure_reasons", [])
            network_reasons = {
                "decoded_fps",
                "dropped_frames",
                "freeze",
                "packet_loss",
                "jitter",
            }
            if not network_reasons.intersection(reasons):
                # Do not mistake decoder/capture instability for a network
                # bitrate limit. Still keep the candidate under observation
                # for the full confirmation window before reporting limited.
                if not self._confirming_stability:
                    print(
                        f"[StreamCalibration] No bitrate change: "
                        f"non-network failure reasons={','.join(reasons) or 'unknown'} "
                        f"lost={metrics.get('packets_lost', 0)}; "
                        f"starting {self.stability_seconds:g}-second confirmation",
                        flush=True,
                    )
                    self._decreasing_bitrate = True
                    self._confirming_stability = True
                    self._reset_stage_locked("confirming")
                    self._write_state_locked(extra={"last_metrics": metrics})
                    return
                selected = self._best_tier or self._active_tier
                self._complete_locked(selected, metrics, "limited")
                return
            if self._confirming_stability:
                # A long confirmation failure invalidates this candidate but
                # preserves lower short-probe successes as the next bracket.
                self._stable_tiers.pop(self._active_tier.target_mbps, None)
                self._best_tier = (
                    self._stable_tiers[max(self._stable_tiers)]
                    if self._stable_tiers
                    else None
                )
            self._confirming_stability = False
            self._decreasing_bitrate = True
            self._search_upper_mbps = min(
                int(self._search_upper_mbps or self._active_tier.target_mbps),
                self._active_tier.target_mbps,
            )
            lower = (
                self._best_tier.target_mbps
                if self._best_tier is not None
                else max(1, min(7, self._active_tier.target_mbps // 2))
            )
            upper = self._search_upper_mbps
            if upper - lower <= _BITRATE_SEARCH_RESOLUTION_MBPS:
                if self._best_tier is None:
                    self._complete_locked(self._active_tier, metrics, "limited")
                    return
                selected = self._best_tier
                if selected != self._active_tier:
                    self._active_tier = selected
                    self._pending_tier = selected
                self._confirming_stability = True
                self._reset_stage_locked("confirming")
                self._write_state_locked(extra={"last_metrics": metrics})
                return
            next_target = (lower + upper) // 2
            print(
                f"[StreamCalibration] Receiver instability detected "
                f"({','.join(reason for reason in reasons if reason in network_reasons)}); "
                f"lost={metrics.get('packets_lost', 0)} "
                f"decoded={metrics.get('decoded_fps', 0)} "
                f"drop_ratio={metrics.get('drop_ratio', 0)} "
                f"freeze={metrics.get('freeze_count', 0)} "
                f"jitter={metrics.get('jitter_buffer_ms', 0)}ms; "
                f"binary range stable={lower}M unstable={upper}M; "
                f"testing {next_target}M",
                flush=True,
            )

        if passed and not self._decreasing_bitrate:
            print(
                f"[StreamCalibration] Probe stable; increasing bitrate "
                f"{self._active_tier.target_mbps}M -> {next_target}M",
                flush=True,
            )

        if next_target < 8:
            selected = self._best_tier or self._active_tier
            self._complete_locked(
                selected,
                metrics,
                "stable" if self._best_tier is not None else "limited",
            )
            return
        if next_target > self._bitrate_limit:
            # The configured ceiling was reached without loss. It is still a
            # final candidate, so require the same long stability confirmation.
            self._decreasing_bitrate = True
            self._confirming_stability = True
            self._reset_stage_locked("confirming")
            self._write_state_locked(extra={"last_metrics": metrics})
            return

        next_tier = CalibrationTier(
            fps=30,
            target_mbps=next_target,
            peak_mbps=round(next_target * 1.15),
        )
        self._tier_index += 1
        self._tiers.append(next_tier)
        self._active_tier = next_tier
        self._pending_tier = next_tier
        self._reset_stage_locked("reconnecting")
        self._write_state_locked(extra={"last_metrics": metrics})

    def _reset_stage_locked(self, status: str) -> None:
        self._receiver_reports.clear()
        self._latest_receiver_report = {}
        self._sender_report = None
        self._sender_reports.clear()
        self._stage_started = None
        self._measurement_started = None
        self._status = status

    def _complete_locked(
        self,
        selected: CalibrationTier,
        metrics: dict[str, Any],
        stability: str,
    ) -> None:
        if selected != self._active_tier:
            self._pending_tier = selected
            self._active_tier = selected
        self._status = "complete"
        network_max_mbps = max(1, int(selected.target_mbps))
        safe_target_mbps = max(1, int(network_max_mbps * _SAFE_TARGET_RATIO))
        if stability == "stable":
            resolution_floor = minimum_bitrate_for_resolution(
                *self._input_resolution
            )
            safe_target_mbps = max(
                safe_target_mbps,
                min(resolution_floor, network_max_mbps),
            )
        safe_target_mbps = min(safe_target_mbps, network_max_mbps)
        safe_peak_mbps = max(
            safe_target_mbps,
            int(network_max_mbps * _SAFE_PEAK_RATIO),
        )
        profile = {
            "version": 4,
            "created_at": time.time(),
            "stream_key": self.stream_key,
            "fps": selected.fps,
            "target_mbps": safe_target_mbps,
            "peak_mbps": safe_peak_mbps,
            "network_max_mbps": network_max_mbps,
            "measured_bitrate_mbps": float(
                metrics.get("received_bitrate_mbps", 0.0) or 0.0
            ),
            "encoded_bitrate_mbps": float(
                metrics.get("encoded_bitrate_mbps", 0.0) or 0.0
            ),
            "stability": stability,
            "metrics": metrics,
            "fingerprint": self.fingerprint,
        }
        self._result = dict(profile)
        self._write_json(self.profile_path, profile)
        self._write_state_locked(extra={"result": profile})
        print(
            f"[StreamCalibration] Complete: {selected.fps} FPS "
            f"network_max={network_max_mbps}M safe_target={safe_target_mbps}M "
            f"safe_peak={safe_peak_mbps}M "
            f"stability={stability}",
            flush=True,
        )

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_locked()

    def _state_locked(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        stage_seconds = (
            self.stability_seconds
            if self._confirming_stability
            else self.stage_seconds
        )
        if self._status == "complete":
            progress = 1.0
        elif self._stage_started is None:
            progress = 0.0
        elif self._measurement_started is None:
            progress = min(
                1.0,
                (self._clock() - self._stage_started)
                / max(0.001, stage_seconds),
            )
        else:
            progress = min(
                1.0,
                (self._clock() - self._measurement_started)
                / max(0.001, stage_seconds),
            )
        overall_progress = (
            1.0
            if self._status == "complete"
            else min(
                1.0,
                (
                    self._completed_stage_count + progress
                )
                / max(1, self._progress_total),
            )
        )
        payload = {
            "status": self._status,
            "tier_index": self._tier_index,
            "tier_count": len(self._tiers),
            "tier": asdict(self._active_tier),
            "stage_progress": progress,
            "overall_progress": overall_progress,
            "completed_stages": self._completed_stage_count,
            "estimated_stages": self._progress_total,
            "receiver_connected": bool(self._latest_receiver_report),
            "receiver_samples": len(self._receiver_reports),
            "receiver_latest": dict(self._latest_receiver_report),
            "sender": dict(self._sender_report or {}),
            "current_test": {
                "target_mbps": int(self._active_tier.target_mbps),
                "peak_mbps": int(self._active_tier.peak_mbps),
                "fps": int(self._active_tier.fps),
                "phase": (
                    "稳定性确认"
                    if self._confirming_stability
                    else "码率测试"
                ),
            },
            "last_metrics": dict(self._last_metrics),
        }
        if self._result is not None:
            payload["result"] = dict(self._result)
        if extra:
            payload.update(extra)
        return payload

    def _write_state(self) -> None:
        with self._lock:
            self._write_state_locked()

    def _write_state_locked(self, extra: dict[str, Any] | None = None) -> None:
        self._write_json(self.state_path, self._state_locked(extra))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _viewer_html(self) -> str:
        stream_port = self.stream_port
        stream_key = json.dumps(self.stream_key)
        whep_url = (
            "`${location.origin}/${encodeURIComponent(streamKey)}/whep`"
            if self.bind_port != self.stream_port
            else f"`http://${{location.hostname}}:{stream_port}/${{encodeURIComponent(streamKey)}}/whep`"
        )
        return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Desktop2Stereo 自动校准</title>
<style>body{{margin:0;background:#090b10;color:#eef;font:16px system-ui;text-align:center}}
main{{max-width:1100px;margin:auto;padding:20px}}video{{width:100%;background:#000;border-radius:10px}}
#state{{margin:12px;padding:10px;background:#182033;border-radius:8px}}.metrics{{font-family:monospace}}</style>
</head><body><main><h2>Desktop2Stereo 自动网络与性能校准</h2>
<div id=\"state\">正在连接测试流…</div><video id=\"video\" autoplay playsinline controls></video>
<p class=\"metrics\" id=\"metrics\"></p><pre id=\"debug\"></pre></main><script>
const streamKey={stream_key}; let pc=null, timer=null, stateTimer=null, previous=null, reconnectTimer=null, sessionUrl=null;
const video=document.getElementById('video'), state=document.getElementById('state'), metrics=document.getElementById('metrics'), debug=document.getElementById('debug');
function setState(message){{state.textContent=message;debug.textContent=`${{new Date().toLocaleTimeString()}} ${{message}}`;}}
function mbps(value){{return Number.isFinite(Number(value))?`${{Number(value).toFixed(1)}} Mbps`:'--';}}
function renderCalibration(payload){{
 const current=payload.current_test||payload.tier||{{}}, last=payload.last_metrics||{{}}, result=payload.result;
 if(payload.status==='complete'&&result){{
  setState(`✅ 校准完成：稳定上限 ${{mbps(result.network_max_mbps)}} · 安全码率 ${{mbps(result.target_mbps)}} · 峰值 ${{mbps(result.peak_mbps)}} · ${{result.fps||0}} FPS`);
  const m=result.metrics||last;
  metrics.textContent=`最终结果：稳定上限 ${{mbps(result.network_max_mbps)}} · 安全目标 ${{mbps(result.target_mbps)}} · 峰值 ${{mbps(result.peak_mbps)}} · 实测接收 ${{mbps(result.measured_bitrate_mbps)}} · 解码 ${{Number(m.decoded_fps||0).toFixed(1)}} FPS`;
 }} else {{
  const progress=Math.round(Number(payload.stage_progress||0)*100);
  if(['testing','settling','reconnecting','confirming'].includes(payload.status))setState(`正在${{current.phase||'码率测试'}}：目标 ${{mbps(current.target_mbps)}}，峰值 ${{mbps(current.peak_mbps)}}，${{current.fps||0}} FPS · 当前阶段 ${{progress}}%`);
  metrics.textContent=`当前测试：目标 ${{mbps(current.target_mbps)}} / 峰值 ${{mbps(current.peak_mbps)}} · 浏览器接收 ${{mbps(last.received_bitrate_mbps)}} · 解码 ${{Number(last.decoded_fps||0).toFixed(1)}} FPS · 丢帧 ${{last.dropped_frames||0}} · 丢包 ${{last.packets_lost||0}}`;
 }}
}}
async function pollCalibrationState(){{try{{const response=await fetch('/state',{{cache:'no-store'}});if(response.ok)renderCalibration(await response.json());}}catch(_err){{}}}}
async function waitIce(p){{if(p.iceGatheringState==='complete')return;await new Promise(resolve=>{{
 const f=()=>{{if(p.iceGatheringState==='complete'){{p.removeEventListener('icegatheringstatechange',f);resolve();}}}};
 p.addEventListener('icegatheringstatechange',f);setTimeout(resolve,3000);}})}}
function reconnect(){{clearInterval(timer);clearInterval(stateTimer);if(sessionUrl)fetch(sessionUrl,{{method:'DELETE'}}).catch(()=>{{}});sessionUrl=null;
 if(pc)pc.close();pc=null;previous=null;clearTimeout(reconnectTimer);
 reconnectTimer=setTimeout(connect,1000);}}
async function connect(){{try{{setState('正在连接测试流…');pc=new RTCPeerConnection();
 pc.addTransceiver('video',{{direction:'recvonly'}});
 pc.ontrack=e=>{{video.srcObject=e.streams&&e.streams[0]?e.streams[0]:new MediaStream([e.track]);video.play().catch(()=>{{}});}};
 pc.oniceconnectionstatechange=()=>{{debug.textContent=`ICE: ${{pc.iceConnectionState}}`;}};
 pc.onconnectionstatechange=()=>{{
  if(pc.connectionState==='connected')setState('已连接，正在进行分档码率测试，请保持页面开启');
  if(['failed','closed','disconnected'].includes(pc.connectionState)){{setState(`WebRTC连接失败: ${{pc.connectionState}}`);reconnect();}}}};
 await pc.setLocalDescription(await pc.createOffer());await waitIce(pc);
 const url={whep_url};
 const response=await fetch(url,{{method:'POST',headers:{{'Content-Type':'application/sdp'}},body:pc.localDescription.sdp}});
 if(!response.ok)throw new Error(`WHEP ${{response.status}} ${{await response.text()}}`);
 const locationHeader=response.headers.get('Location');if(locationHeader)sessionUrl=new URL(locationHeader,url).href;
 await pc.setRemoteDescription({{type:'answer',sdp:await response.text()}});timer=setInterval(report,1000);stateTimer=setInterval(pollCalibrationState,1000);pollCalibrationState();
 }}catch(err){{setState('等待测试流：'+err.message);reconnect();}}}}
async function report(){{if(!pc)return;const stats=await pc.getStats();let inbound=null;
 stats.forEach(s=>{{if(s.type==='inbound-rtp'&&s.kind==='video')inbound=s;}});if(!inbound)return;
 const now={{time:Date.now()/1000,framesDecoded:inbound.framesDecoded||0,framesDropped:inbound.framesDropped||0,
  packetsReceived:inbound.packetsReceived||0,
  packetsLost:inbound.packetsLost||0,bytesReceived:inbound.bytesReceived||0,freezeCount:inbound.freezeCount||0,
  jitterBufferDelay:inbound.jitterBufferDelay||0,jitterBufferEmittedCount:inbound.jitterBufferEmittedCount||0}};
 if(previous){{const dt=Math.max(.1,now.time-previous.time), emitted=now.jitterBufferEmittedCount-previous.jitterBufferEmittedCount;
  const payload={{decoded_fps:(now.framesDecoded-previous.framesDecoded)/dt,dropped_frames:now.framesDropped-previous.framesDropped,
  packets_lost:now.packetsLost-previous.packetsLost,packets_received:now.packetsReceived-previous.packetsReceived,
   freeze_count:now.freezeCount-previous.freezeCount,
   bitrate_mbps:(now.bytesReceived-previous.bytesReceived)*8/dt/1e6,
   jitter_buffer_ms:emitted>0?(now.jitterBufferDelay-previous.jitterBufferDelay)/emitted*1000:0,
   width:inbound.frameWidth||0,height:inbound.frameHeight||0}};
  metrics.textContent=`当前浏览器：解码 ${{payload.decoded_fps.toFixed(1)}} FPS · 接收 ${{payload.bitrate_mbps.toFixed(1)}} Mbps · 丢帧 ${{payload.dropped_frames}} · 丢包 ${{payload.packets_lost}}`;
  fetch('/stats',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}}).catch(()=>{{}});
 }}previous=now;}}connect();
</script></body></html>"""
