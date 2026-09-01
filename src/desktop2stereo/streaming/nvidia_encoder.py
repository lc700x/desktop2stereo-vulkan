from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import subprocess
import threading
from typing import Any


def rgb_cuda_to_nv12(frame: Any) -> tuple[Any, Any]:
    """Convert a CUDA RGB tensor to NV12 planes without downloading to CPU."""
    import torch

    image = frame.detach()
    if not bool(getattr(image, "is_cuda", False)):
        raise ValueError("PyNvVideoCodec GPU input requires a CUDA tensor")
    if image.ndim == 4:
        if int(image.shape[0]) != 1:
            raise ValueError(f"expected one frame, got {tuple(image.shape)!r}")
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected RGB HWC or CHW tensor, got {tuple(image.shape)!r}")
    if int(image.shape[0]) in (1, 3, 4):
        image = image[:3].permute(1, 2, 0)
    elif int(image.shape[-1]) in (1, 3, 4):
        image = image[..., :3]
    else:
        raise ValueError(f"unsupported RGB tensor shape: {tuple(image.shape)!r}")
    height, width = (int(image.shape[0]), int(image.shape[1]))
    if height % 2 or width % 2:
        raise ValueError("NV12 requires even width and height")
    if image.dtype == torch.uint8:
        rgb = image.float()
    else:
        # Runtime floating-point SBS output is normalized to 0..1.
        rgb = image.float().clamp(0.0, 1.0) * 255.0
    r, g, b = rgb.unbind(dim=-1)
    y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0).clamp(0, 255).to(torch.uint8)
    u = (-0.148 * r - 0.291 * g + 0.439 * b + 128.0).clamp(0, 255)
    v = (0.439 * r - 0.368 * g - 0.071 * b + 128.0).clamp(0, 255)
    u = u.reshape(height // 2, 2, width // 2, 2).mean(dim=(1, 3))
    v = v.reshape(height // 2, 2, width // 2, 2).mean(dim=(1, 3))
    uv = torch.stack((u, v), dim=-1).reshape(height // 2, width).to(torch.uint8)
    nv12 = torch.empty((height * 3 // 2, width, 1), dtype=torch.uint8, device=image.device)
    y_plane = nv12[:height]
    uv_plane = nv12[height:].reshape(height // 2, width // 2, 2)
    y_plane.copy_(y.unsqueeze(-1))
    uv_plane.copy_(uv.reshape(height // 2, width // 2, 2))
    # PyNvVideoCodec owns a separate CUDA/NVENC submission queue and cannot
    # infer a dependency on PyTorch's current stream. Without this hand-off
    # wait it may read the NV12 planes while conversion is still in flight,
    # producing an initialized encoder with corrupted frames.
    torch.cuda.current_stream(device=image.device).synchronize()
    return y_plane, uv_plane


@dataclass
class Nv12CudaFrame:
    """CUDA Array Interface adapter expected by PyNvVideoCodec GPU input."""

    y: Any
    uv: Any

    def cuda(self) -> list[Any]:
        return [self.y, self.uv]


class PyNvVideoCodecEncoder:
    """Small encoder adapter; muxing and transport remain outside this class."""

    def __init__(
        self,
        nvc: Any,
        width: int,
        height: int,
        *,
        hevc: bool,
        fps: int,
        bitrate: int,
        gpu_id: int = 0,
    ):
        codec = "hevc" if hevc else "h264"
        frame_rate = max(1, int(fps))
        target_bitrate = max(1, int(bitrate))
        self._encoder = nvc.CreateEncoder(
            int(width),
            int(height),
            "NV12",
            False,
            codec=codec,
            fps=frame_rate,
            bitrate=target_bitrate,
            maxbitrate=max(target_bitrate, int(target_bitrate * 1.2)),
            gpu_id=max(0, int(gpu_id)),
            tuning_info="ultra_low_latency",
            preset="P1",
            rc="cbr",
            gop=frame_rate,
            idrperiod=frame_rate,
            # WebRTC low-latency streams must remain IPPP. B-frames require
            # display reordering and can make a slow browser reader lose the
            # reference chain when MediaMTX has to discard queued packets.
            bf=0,
            repeatspspps=1,
        )

    def encode(self, rgb_cuda: Any) -> bytes:
        y, uv = rgb_cuda_to_nv12(rgb_cuda)
        packet = self._encoder.Encode(Nv12CudaFrame(y, uv))
        return self._packet_bytes(packet)

    def flush(self) -> bytes:
        return self._packet_bytes(self._encoder.EndEncode())

    @staticmethod
    def _packet_bytes(packets: Any) -> bytes:
        if packets is None:
            return b""
        if isinstance(packets, (bytes, bytearray, memoryview)):
            return bytes(packets)
        output = bytearray()
        for packet in packets:
            if isinstance(packet, dict):
                packet = packet.get("data", b"")
            output.extend(bytes(packet))
        return bytes(output)


class H264MpegTsTimestampMuxer:
    """Wrap timestamped H.264 access units in MPEG-TS for FFmpeg input."""

    VIDEO_PID = 0x101
    PMT_PID = 0x100
    PROGRAM = 1

    def __init__(self, *, fps: int) -> None:
        self.fps = max(1, int(fps))
        self._continuity = {0: 0, self.PMT_PID: 0, self.VIDEO_PID: 0}
        self._started = False

    @staticmethod
    def _crc32(data: bytes) -> int:
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte << 24
            for _ in range(8):
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
        return crc

    @staticmethod
    def _pts_bytes(prefix: int, value: int) -> bytes:
        value = max(0, int(value)) & ((1 << 33) - 1)
        return bytes([
            prefix | ((value >> 30) & 0x07) << 1 | 1,
            (value >> 22) & 0xFF,
            ((value >> 15) & 0x7F) << 1 | 1,
            (value >> 7) & 0xFF,
            (value & 0x7F) << 1 | 1,
        ])

    def _section_packet(self, pid: int, section: bytes) -> bytes:
        packet = self._ts_header(pid, payload_start=True)
        payload = bytes([0]) + section
        return packet + payload + bytes(188 - len(packet) - len(payload))

    def _ts_header(self, pid: int, *, payload_start: bool, adaptation: bytes | None = None) -> bytes:
        cc = self._continuity[pid]
        self._continuity[pid] = (cc + 1) & 0x0F
        if adaptation is None:
            control = 1
        else:
            control = 3
        return bytes([
            0x47,
            (0x40 if payload_start else 0) | ((pid >> 8) & 0x1F),
            pid & 0xFF,
            (control << 4) | cc,
        ])

    def _pat(self) -> bytes:
        body = bytes([0x00, 0x01, 0xC1, 0x00, 0x00, 0xE0 | (self.PMT_PID >> 8), self.PMT_PID & 0xFF])
        section = bytes([0x00, 0xB0, len(body) + 4]) + body
        return self._section_packet(0, section + self._crc32(section).to_bytes(4, "big"))

    def _pmt(self) -> bytes:
        body = bytes([
            0x00, 0x01, 0xC1, 0x00, 0x00,
            0xE0 | (self.VIDEO_PID >> 8), self.VIDEO_PID & 0xFF,
            0xF0, 0x00,
            0x1B, 0xE0 | (self.VIDEO_PID >> 8), self.VIDEO_PID & 0xFF, 0xF0, 0x00,
        ])
        section = bytes([0x02, 0xB0, len(body) + 4]) + body
        return self._section_packet(self.PMT_PID, section + self._crc32(section).to_bytes(4, "big"))

    def _pes(self, data: bytes, pts: int, dts: int, duration: int) -> bytes:
        pts90 = max(0, int(pts)) * 90000 // self.fps
        dts90 = max(0, int(dts)) * 90000 // self.fps
        pts_dts = self._pts_bytes(0x30 if pts90 != dts90 else 0x20, pts90)
        if pts90 != dts90:
            pts_dts += self._pts_bytes(0x10, dts90)
        header = b"\x00\x00\x01\xE0\x00\x00\x80\x00" + bytes([len(pts_dts)]) + pts_dts
        return header + data

    def _packetize(self, payload: bytes, pid: int, *, pcr90: int | None = None) -> bytes:
        output = bytearray()
        offset = 0
        first = True
        while offset < len(payload):
            remaining = len(payload) - offset
            include_pcr = first and pcr90 is not None
            max_payload = 176 if include_pcr else 184
            chunk = min(remaining, max_payload)
            needs_adaptation = include_pcr or chunk < 184
            adaptation = None
            if needs_adaptation:
                pcr_bytes = b""
                flags = 0
                if include_pcr:
                    pcr = max(0, int(pcr90)) * 300
                    pcr_bytes = bytes([
                        (pcr >> 25) & 0xFF, (pcr >> 17) & 0xFF,
                        (pcr >> 9) & 0xFF, (pcr >> 1) & 0xFF,
                        ((pcr & 1) << 7) | 0x7E, 0x00,
                    ])
                    flags = 0x10
                adaptation_length = 183 - chunk
                if adaptation_length < 1 or (include_pcr and adaptation_length < 7):
                    raise RuntimeError("invalid MPEG-TS adaptation field size")
                stuffing = adaptation_length - 1 - len(pcr_bytes)
                adaptation = bytes([adaptation_length, flags]) + pcr_bytes + bytes([0xFF]) * stuffing
            output.extend(self._ts_header(pid, payload_start=first, adaptation=adaptation))
            if adaptation is not None:
                output.extend(adaptation)
            output.extend(payload[offset:offset + chunk])
            offset += chunk
            first = False
        return bytes(output)

    def wrap(self, packet: Any) -> bytes:
        data = bytes(packet.data)
        if not data:
            return b""
        pts90 = max(0, int(packet.pts)) * 90000 // self.fps
        output = bytearray()
        if not self._started:
            output.extend(self._pat())
            output.extend(self._pmt())
            self._started = True
        output.extend(self._packetize(self._pes(data, packet.pts, packet.dts, packet.duration), self.VIDEO_PID, pcr90=pts90))
        return bytes(output)


class PyNvSrtVideoOutput:
    """Mux PyNvVideoCodec packets with optional PCM audio through FFmpeg."""

    def __init__(
        self,
        encoder: Any,
        ffmpeg_path: str,
        srt_url: str | None = None,
        *,
        codec: str = "h264",
        fps: int = 30,
        audio_url: str | None = None,
        audio_delay: float = 0.0,
        audio_codec: str = "libopus",
        output_args: list[str] | None = None,
        creationflags: int = 0,
    ):
        self.encoder = encoder
        self._timestamped_input = callable(getattr(encoder, "encode_timed", None))
        self._timestamp_muxer = (
            H264MpegTsTimestampMuxer(fps=fps)
            if self._timestamped_input
            else None
        )
        if output_args is None:
            if not srt_url:
                raise ValueError("srt_url or output_args is required")
            output_args = ["-f", "mpegts", srt_url]
        command = [
            str(ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            # PyNv packets are Annex-B bytes without AVPacket PTS/DTS.
            # Generate a monotonic video timeline before RTSP interleaving
            # with the live Opus input.
            "-fflags",
            "+nobuffer+genpts",
            "-flags",
            "low_delay",
            "-probesize",
            "64",
            "-analyzeduration",
            "0",
            # The PyNv pipe carries Annex-B bytes only; bind each demuxed
            # video packet to the live clock so RTSP never receives stream 0
            # packets without PTS/DTS.
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "mpegts" if self._timestamped_input else (
                "hevc" if codec.casefold() in {"hevc", "h265"} else "h264"
            ),
            *([] if self._timestamped_input else ["-r", str(max(1, int(fps)))]),
            "-i",
            "pipe:0",
        ]
        if audio_url:
            command.extend(
                [
                    "-thread_queue_size",
                    "1024",
                    "-itsoffset",
                    str(float(audio_delay)),
                    "-f",
                    "s16le",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-i",
                    audio_url,
                ]
            )
        command.extend(["-map", "0:v:0"])
        if audio_url:
            command.extend(["-map", "1:a:0"])
        command.extend(["-c:v", "copy", "-fps_mode", "cfr"])
        if audio_url:
            if audio_codec == "libopus":
                # Same timeline strategy as the shared FFmpeg path (the
                # Windows soundcard fix): the s16le/UDP audio input must not
                # use the demuxer -use_wallclock_as_timestamps option (it
                # silences the whole audio chain on the bundled FFmpeg build),
                # so the audio PTS are re-anchored to the same wall-clock base
                # as the video input with asetpts=RTCTIME. Order matters:
                # asetpts must run BEFORE aresample (asetpts after aresample
                # and asetpts alone both yield an empty audio stream). The
                # -itsoffset audio delay is folded into the RTCTIME offset
                # because asetpts overwrites the demuxer PTS that the offset
                # shifted.
                delay_us = int(round(float(audio_delay) * 1e6))
                command.extend(
                    [
                        "-af",
                        f"asetpts=RTCTIME{delay_us:+d},aresample=async=1",
                        "-c:a",
                        "libopus",
                        "-ar",
                        "48000",
                        "-ac",
                        "2",
                        "-b:a",
                        "96k",
                    ]
                )
            else:
                command.extend(["-c:a", "aac", "-ar", "48000", "-b:a", "128k"])
        else:
            command.append("-an")
        command.extend(
            [
                "-muxdelay",
                "0",
                "-muxpreload",
                "0",
                "-flush_packets",
                "1",
                "-max_interleave_delta",
                "100000",
                *output_args,
            ]
        )
        self.command = command
        self._stderr_tail = deque(maxlen=40)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=int(creationflags),
        )
        self._stderr_thread = None
        stderr = getattr(self.process, "stderr", None)
        if stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(stderr,),
                name="NativeNvencMuxLog",
                daemon=True,
            )
            self._stderr_thread.start()
        print(
            "[DirectSbsStream] NativeNVENC mux active: "
            f"audio={'enabled' if audio_url else 'disabled'} "
            f"audio_codec={audio_codec if audio_url else 'none'} "
            f"audio_input={audio_url or 'none'}",
            flush=True,
        )

    def _drain_stderr(self, stderr) -> None:
        try:
            for raw_line in iter(stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self._stderr_tail.append(line)
                print(f"[DirectSbsStream] NativeNVENC mux: {line}", flush=True)
        except (OSError, ValueError):
            return

    def submit_cuda_frame(self, frame: Any) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(
                f"NVENC packet muxer exited with code {self.process.returncode}"
            )
        if self.process.stdin is None:
            raise RuntimeError("NVENC packet muxer stdin is unavailable")
        if self._timestamped_input:
            for packet in self.encoder.encode_timed(frame):
                transport_packet = self._timestamp_muxer.wrap(packet)
                if transport_packet:
                    self.process.stdin.write(transport_packet)
        else:
            packet = self.encoder.encode(frame)
            if packet:
                self.process.stdin.write(packet)
        self.process.stdin.flush()

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                if self._timestamped_input:
                    for packet in self.encoder.flush_timed():
                        transport_packet = self._timestamp_muxer.wrap(packet)
                        if transport_packet and self.process.stdin is not None:
                            self.process.stdin.write(transport_packet)
                else:
                    tail = self.encoder.flush()
                    if tail and self.process.stdin is not None:
                        self.process.stdin.write(tail)
                if self.process.stdin is not None:
                    self.process.stdin.flush()
        finally:
            if self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=3.0)
            close_encoder = getattr(self.encoder, "close", None)
            if callable(close_encoder):
                close_encoder()
