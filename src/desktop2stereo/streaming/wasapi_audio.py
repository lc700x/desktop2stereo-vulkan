from __future__ import annotations

import os
import socket
import threading
import time
import warnings
from typing import Any

import numpy as np


def query_soundcard_loopback_devices() -> list[str] | None:
    """Return loopback speakers with the Windows default speaker first."""
    try:
        import soundcard as sc
        default_speaker = sc.default_speaker()
        all_speakers = list(sc.all_speakers())
        ordered_speakers = []
        if default_speaker is not None:
            ordered_speakers.append(default_speaker)
        ordered_speakers.extend(all_speakers)

        speakers = []
        seen_ids = set()
        for speaker in ordered_speakers:
            speaker_id = str(getattr(speaker, "id", "") or "")
            if speaker_id and speaker_id in seen_ids:
                continue
            if speaker_id:
                seen_ids.add(speaker_id)
            loopback = sc.get_microphone(
                id=speaker.id,
                include_loopback=True,
            )
            if loopback is not None and bool(getattr(loopback, "isloopback", False)):
                name = str(getattr(speaker, "name", "") or "").strip()
                if name and name not in speakers:
                    speakers.append(name)
        return speakers or None
    except Exception:
        return None


class SoundcardLoopbackSender:
    """Capture the Windows default speaker loopback and send PCM over localhost."""

    def __init__(self, device_name: str | None = None, *, samplerate: int = 48000):
        import soundcard as sc

        self.samplerate = int(samplerate)
        self.channels = 2
        # 1024 frames is only 21.3 ms at 48 kHz and is easily overrun while
        # the GPU stream is starting. Keep the block size configurable, with a
        # safer default that still stays below 100 ms of audio latency.
        self.blocksize = max(
            1024,
            int(os.environ.get("D2S_WASAPI_BLOCKSIZE", "4096")),
        )
        # Keep each localhost UDP datagram below the normal Ethernet MTU.
        # A 4096-frame stereo s16le block is about 16 KiB and can fragment;
        # fragmentation is especially harmful when the video muxer bursts.
        self.udp_frames = max(
            240,
            int(os.environ.get("D2S_WASAPI_UDP_FRAMES", "240")),
        )
        self._soundcard = sc
        self._loopback = self._resolve_loopback(device_name)
        self.device_name = str(getattr(self._loopback, "name", "") or device_name or "default")
        self._packet_count = 0
        self._discontinuity_count = 0
        self._silent_packet_count = 0
        reservation = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reservation.bind(("127.0.0.1", 0))
        self.port = int(reservation.getsockname()[1])
        reservation.close()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._startup_done = threading.Event()
        self._startup_error: Exception | None = None
        self._runtime_error: Exception | None = None
        self._thread: threading.Thread | None = None

    def _resolve_loopback(self, device_name: str | None) -> Any:
        requested = str(device_name or "").strip()
        speaker = None
        if requested:
            for candidate in self._soundcard.all_speakers():
                if str(getattr(candidate, "name", "")) == requested:
                    speaker = candidate
                    break
        if speaker is None:
            speaker = self._soundcard.default_speaker()
        if speaker is None:
            raise RuntimeError("No Windows default speaker is available")
        loopback = self._soundcard.get_microphone(
            id=speaker.id,
            include_loopback=True,
        )
        if loopback is None or not bool(getattr(loopback, "isloopback", False)):
            raise RuntimeError(
                f"No WASAPI loopback endpoint is available for speaker {speaker.name!r}"
            )
        return loopback

    @property
    def ffmpeg_url(self) -> str:
        return f"udp://127.0.0.1:{self.port}?fifo_size=65536&overrun_nonfatal=1"

    def start(self, *, timeout: float = 2.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="WasapiLoopback", daemon=True)
        self._thread.start()
        if not self._startup_done.wait(max(0.1, float(timeout))):
            self.close()
            raise RuntimeError("Windows loopback capture produced no PCM data")
        if self._startup_error is not None:
            error = self._startup_error
            self.close()
            raise RuntimeError(
                f"Windows loopback capture failed: {type(error).__name__}: {error}"
            ) from error
        print(
            "[DirectSbsStream] WASAPI loopback active: "
            f"speaker={self.device_name!r} samplerate={self.samplerate} "
            f"channels={self.channels} blocksize={self.blocksize} "
            f"udp_frames={self.udp_frames} udp={self.port}",
            flush=True,
        )

    def _run(self) -> None:
        produced_pcm = False
        status_interval = max(1, int(os.environ.get("D2S_WASAPI_STATUS_SECS", "5")))
        last_status = time.monotonic()
        try:
            with self._loopback.recorder(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
            ) as recorder:
                while not self._stop.is_set():
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        samples = recorder.record(numframes=self.blocksize)
                    for warning in captured:
                        if "discontinuity" not in str(warning.message).casefold():
                            continue
                        self._discontinuity_count += 1
                        if self._discontinuity_count == 1 or self._discontinuity_count % 10 == 0:
                            print(
                                "[DirectSbsStream] WASAPI recording discontinuity: "
                                f"count={self._discontinuity_count} "
                                f"blocksize={self.blocksize}; continuing capture",
                                flush=True,
                            )
                    pcm = np.asarray(samples, dtype=np.float32)
                    pcm = np.clip(pcm, -1.0, 1.0)
                    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
                    if peak < 1e-5:
                        self._silent_packet_count += 1
                    elif self._silent_packet_count == self._packet_count:
                        print(
                            "[DirectSbsStream] WASAPI capture is live: "
                            f"first non-silent block (peak={peak:.4f})",
                            flush=True,
                        )
                    pcm_bytes = (pcm * 32767.0).astype(np.int16).tobytes()
                    packet_bytes = self.udp_frames * self.channels * 2
                    for offset in range(0, len(pcm_bytes), packet_bytes):
                        self._socket.sendto(
                            pcm_bytes[offset : offset + packet_bytes],
                            ("127.0.0.1", self.port),
                        )
                    self._packet_count += 1
                    # Keep periodic PCM level counters out of normal logs.
                    # Startup and discontinuity diagnostics remain visible.
                    produced_pcm = True
                    self._startup_done.set()
                    now = time.monotonic()
                    if now - last_status >= status_interval:
                        last_status = now
                        print(
                            "[DirectSbsStream] WASAPI capture status: "
                            f"speaker={self.device_name!r} packets={self._packet_count} "
                            f"silent={self._silent_packet_count} "
                            f"peak={peak:.4f} "
                            f"discontinuities={self._discontinuity_count}",
                            flush=True,
                        )
        except Exception as exc:
            if self._stop.is_set():
                return
            if not produced_pcm:
                self._startup_error = exc
                self._startup_done.set()
                self._stop.set()
                return
            self._runtime_error = exc
            print(
                "[DirectSbsStream] Windows loopback capture interrupted: "
                f"{type(exc).__name__}: {exc}; continuing with silent audio",
                flush=True,
            )
            self._send_silence_until_stopped()

    def _send_silence_until_stopped(self) -> None:
        """Keep FFmpeg's mapped audio input alive after a capture interruption."""
        frames_per_packet = 1024
        packet = np.zeros(
            frames_per_packet * self.channels,
            dtype=np.int16,
        ).tobytes()
        interval = frames_per_packet / float(self.samplerate)
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                packet_bytes = self.udp_frames * self.channels * 2
                for offset in range(0, len(packet), packet_bytes):
                    self._socket.sendto(
                        packet[offset : offset + packet_bytes],
                        ("127.0.0.1", self.port),
                    )
            except OSError:
                if not self._stop.is_set():
                    raise
                return
            deadline += interval
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._socket.close()
