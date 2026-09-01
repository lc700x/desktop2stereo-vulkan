from __future__ import annotations

import importlib
import math
import os
import re
from pathlib import Path
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Tuple
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np

from streaming.encoder_profile import EncoderProfile
from streaming.mjpeg_streamer import MJPEGStreamer
from streaming.runtime_manager import ensure_runtime
from streaming.nvidia_encoder import PyNvSrtVideoOutput, PyNvVideoCodecEncoder
from streaming.native_rtsp_output import NativeRtspAvOutput
from streaming.nvenc_cudaarray_bridge import NvencCudaArrayEncoder
from streaming.stream_calibration import StreamCalibrationController
from streaming.wasapi_audio import SoundcardLoopbackSender
from streaming.vulkan_capabilities import probe_vulkan_video
from streaming.vulkan_bridge import VulkanNativeBridge
from streaming.opengl_stream_backend import OpenGLFallbackBackend
from streaming.aspect import (
    apply_aspect_on_cpu,
    apply_aspect_on_gpu,
    normalize_display_fit_mode,
    transport_canvas_size,
)


_PYNVVIDEO_CODEC = None
_PYNVVIDEO_CODEC_ERROR: str | None = None
_PYNVVIDEO_DLL_HANDLES: list[Any] = []


def _load_pynvvideo_codec() -> Any | None:
    """Load PyNvVideoCodec with bundled CUDA runtime DLLs when available."""
    global _PYNVVIDEO_CODEC, _PYNVVIDEO_CODEC_ERROR
    if _PYNVVIDEO_CODEC is not None:
        return _PYNVVIDEO_CODEC
    if _PYNVVIDEO_CODEC_ERROR is not None:
        return None
    try:
        if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
            candidates = []
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                candidates.append(Path(cuda_path) / "bin")
            candidates.extend(
                [
                    Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
                    Path(__file__).resolve().parents[2] / "python3" / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
                ]
            )
            for path in candidates:
                if path.is_dir():
                    _PYNVVIDEO_DLL_HANDLES.append(os.add_dll_directory(str(path)))
        _PYNVVIDEO_CODEC = importlib.import_module("PyNvVideoCodec")
        return _PYNVVIDEO_CODEC
    except Exception as exc:
        _PYNVVIDEO_CODEC_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _kill_process_on_port(port: int, proto: str = "tcp") -> None:
    """Kill process occupying given port (best-effort, Windows)."""
    try:
        # Use netstat to find PID
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform.system() == "Windows" else 0
        proto_flag = "-p tcp" if proto == "tcp" else "-p udp"
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=3.0, creationflags=creationflags,
        )
        target = f":{port}"
        for line in result.stdout.splitlines():
            if target not in line:
                continue
            # Filter by proto
            if proto == "tcp" and "TCP" not in line:
                continue
            if proto == "udp" and "UDP" not in line:
                continue
            parts = line.strip().split()
            if not parts:
                continue
            pid = parts[-1]
            if not pid.isdigit():
                continue
            if int(pid) <= 4:
                continue
            # The MJPEG server binds its port in the streamer constructor, so
            # netstat may list OUR OWN pid on this port. Never kill self.
            if int(pid) == os.getpid():
                continue
            try:
                subprocess.run(["taskkill", "/F", "/PID", pid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2.0, creationflags=creationflags)
            except Exception:
                pass
    except Exception:
        pass

def _kill_orphan_mediamtx_processes(ports: list[int] | None = None) -> None:
    """Kill orphan MediaMTX / FFmpeg processes that hold streaming ports."""
    if ports is None:
        ports = [9998, 9997, 9999, 8000, 8001, 8189, 8554, 1935, 8888, 8889, 8890]
    # 1) Kill by image name (fast path) - only mediamtx/ffmpeg under streaming rtmp
    for img in ["mediamtx.exe", "ffmpeg.exe"]:
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform.system() == "Windows" else 0
            # Use tasklist to check existence first to reduce noise
            subprocess.run(["taskkill", "/F", "/IM", img], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3.0, creationflags=creationflags)
        except Exception:
            pass
    # 2) Kill by port for cases where exe renamed or multiple versions
    for p in ports:
        # try tcp and udp both for 8000/8189 which use udp
        for proto in ("tcp", "udp"):
            _kill_process_on_port(p, proto)
    # give OS time to release
    time.sleep(0.15)


def _darwin_safe_mediamtx_config(config_path: Path) -> Path:
    """Return a runtime-generated MediaMTX config that runs on macOS.

    MediaMTX sets udpReadBufferSize via a socket option that is unimplemented
    on macOS; any non-zero value makes it abort at startup ("read buffer size
    is unimplemented on the current operating system") and close every
    listener. Rewrite it to the OS default (0) in a sibling file and point the
    launch at that copy. Windows/Linux configs are untouched.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        return config_path
    if re.search(r"(?m)^udpReadBufferSize:\s*[1-9]", text) is None:
        return config_path  # already at the OS default
    patched = re.sub(
        r"(?m)^(udpReadBufferSize:\s*)[0-9]+",
        r"\g<1>0",
        text,
    )
    target = config_path.with_name("mediamtx.macos.yml")
    try:
        target.write_text(patched, encoding="utf-8")
    except OSError:
        return config_path
    return target


def _list_darwin_audio_devices(ffmpeg_path: Path) -> list[tuple[int, str]]:
    """Return ``(index, name)`` pairs for every AVFoundation audio device.

    Runs ``ffmpeg -f avfoundation -list_devices true -i ""`` (the reference
    v2.5.0 macOS discovery command) and parses the stderr device listing.
    macOS only; returns [] on any error.
    """
    try:
        result = subprocess.run(
            [
                str(ffmpeg_path),
                "-f",
                "avfoundation",
                "-list_devices",
                "true",
                "-i",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    output = result.stderr or ""
    in_audio = False
    devices: list[tuple[int, str]] = []
    for line in output.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio = True
            continue
        if "AVFoundation video devices:" in line:
            in_audio = False
            continue
        if in_audio:
            match = re.search(r"\[(\d+)\]\s*(.+)", line)
            if match:
                devices.append((int(match.group(1)), match.group(2).strip()))
    return devices


def _auto_select_darwin_audio(ffmpeg_path: Path) -> str:
    """Pick an AVFoundation audio device for loopback capture on macOS.

    Returns the index of the first audio device, preferring loopback-style
    names (BlackHole / Loopback / Virtual / Stereo Mix) so the stream always
    carries sound even when no Stereo Mix device was configured. Returns ""
    when no audio device exists.
    """
    devices = _list_darwin_audio_devices(ffmpeg_path)
    for index, name in devices:
        lowered = name.lower()
        if any(
            token in lowered
            for token in ("blackhole", "loopback", "virtual", "stereo mix")
        ):
            return str(index)
    return str(devices[0][0]) if devices else ""


def _auto_select_windows_audio(ffmpeg_path: Path) -> str:
    """Pick a dshow audio capture device for loopback on Windows.

    Prefers loopback-style names (Stereo Mix / virtual-audio-capturer / What
    U Hear). Returns "" when no loopback-capable device exists so the caller
    runs video-only instead of handing FFmpeg an empty ``-i audio=`` name
    (which fails with ``Error opening input: I/O error`` and kills the
    stream). This is the Windows counterpart of the macOS auto-selection.
    Only loopback-style devices are returned on purpose: auto-selecting an
    arbitrary microphone would broadcast the room instead of system audio.
    """
    try:
        from streaming.audio import (
            find_loopback_audio_devices,
            query_ffmpeg_dshow_audio_devices,
        )

        devices = query_ffmpeg_dshow_audio_devices(ffmpeg_path) or []
        if not devices:
            return ""
        loopback = find_loopback_audio_devices(devices)
        return loopback[0] if loopback else ""
    except Exception:
        return ""


def _required_h264_level(width: int, height: int, fps: int) -> float:
    """Return the smallest H.264 level supporting width x height @ fps.

    The encoder's auto-selected level only accounts for resolution, not the
    frame rate: AMF picks level 5.1 for 4K, but level 5.1 caps 4K at ~30 fps
    (MaxMBPS 983,040 / 32,400 MB per frame). At 4K@40 that SPS is invalid, so
    browser WebRTC decoders reject the stream (black frame) even though RTP
    flows. Compute the level from the actual MB/s load instead:
    MaxMBPS per level (H.264 Table A-1) -> smallest level whose budget fits.
    """
    macroblocks_per_frame = max(1, math.ceil(width / 16)) * max(
        1, math.ceil(height / 16)
    )
    required_mbps = macroblocks_per_frame * max(1, int(fps))
    # (level, MaxMBPS) pairs from the H.264 spec.
    level_budgets = [
        (1.0, 1485), (1.1, 3000), (1.2, 6000), (1.3, 11880),
        (2.0, 11880), (2.1, 19800), (2.2, 20250),
        (3.0, 40500), (3.1, 108000), (3.2, 216000),
        (4.0, 245760), (4.1, 245760), (4.2, 522240),
        (5.0, 589824), (5.1, 983040), (5.2, 2073600),
        (6.0, 4177920), (6.1, 8355840), (6.2, 16711680),
    ]
    for level, budget in level_budgets:
        if required_mbps <= budget:
            return level
    return 6.2


def _format_h264_level(level: float) -> str:
    """Format a numeric H.264 level (e.g. 5.2) as FFmpeg expects ("5.2")."""
    return f"{level:g}"


def runtime_sbs_to_rgb(frame_or_result: Any) -> np.ndarray:
    """Convert a packed SBS runtime tensor/array to contiguous RGB8 HWC."""
    frame = getattr(frame_or_result, "sbs", frame_or_result)
    if frame is None:
        raise ValueError("runtime result does not contain an SBS frame")
    image = frame.detach() if hasattr(frame, "detach") else frame
    # Accelerator tensors (CUDA, MPS, ...) must be copied to host memory
    # before numpy conversion; an is_cuda-only check lets MPS frames through
    # and numpy raises "can't convert mps:0 device type tensor to numpy".
    if hasattr(image, "device") and getattr(image.device, "type", "cpu") != "cpu":
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"unsupported SBS frame shape: {image.shape!r}")
    if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"unsupported SBS channel count: {image.shape[-1]}")
    if image.dtype != np.uint8:
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


class RuntimeSbsRgbConverter:
    """Convert runtime SBS frames with a reusable pinned CUDA download buffer."""

    def __init__(
        self,
        *,
        copy_output: bool = False,
        display_mode: str = "Half-SBS",
        fit_mode: str = "contain",
        input_size: Tuple[int, int] | None = None,
    ) -> None:
        self.copy_output = bool(copy_output)
        self.display_mode = str(display_mode or "Half-SBS").strip()
        self.fit_mode = str(fit_mode or "contain").strip()
        self.input_size = input_size
        self._host_rgb = None

    def convert(self, frame_or_result: Any) -> np.ndarray:
        frame = getattr(frame_or_result, "sbs", frame_or_result)
        if frame is None:
            raise ValueError("runtime result does not contain an SBS frame")
        image = frame.detach() if hasattr(frame, "detach") else frame
        is_cuda = bool(getattr(image, "is_cuda", False))
        # Determine source size before any conversion for aspect handling
        # image may be [1,3,H,W] or [H,W,3] or [3,H,W]
        def _hw_of(t):
            if t.ndim == 4:
                if int(t.shape[-1]) in (1, 3, 4):
                    return int(t.shape[-3]), int(t.shape[-2])  # B,H,W,C
                return int(t.shape[-2]), int(t.shape[-1])  # B,C,H,W
            if t.ndim == 3:
                if int(t.shape[0]) in (1, 3, 4) and int(t.shape[-1]) not in (1, 3, 4):
                    return int(t.shape[-2]), int(t.shape[-1])  # C,H,W
                return int(t.shape[0]), int(t.shape[1])  # H,W,C
            return 0, 0
        h, w = _hw_of(image)
        # Transport canvas mirrors local viewer presentation / legacy
        # fill_16_9: contain pads each eye to a 16:9 canvas before packing;
        # cover/stretch keep the original input aspect.
        tw, th = transport_canvas_size(
            (w, h),
            self.fit_mode,
            input_size=self.input_size,
            display_mode=self.display_mode,
        )
        process_aspect = normalize_display_fit_mode(self.fit_mode) == "contain"
        # For GPU path, keep on GPU and apply per-eye aspect before download.
        # The 16:9 canvas processing is GPU-mandatory for CUDA frames: a
        # failure must surface, never silently degrade to a CPU download that
        # is both slower and skips the aspect requirement.
        if is_cuda:
            if process_aspect:
                # Keep aspect on GPU: convert to HWC uint8 CUDA first, apply, then download
                import torch
                gpu_img = image
                # Normalize to HWC uint8 CUDA for aspect func
                if gpu_img.ndim == 4:
                    gpu_img = gpu_img[0]
                if gpu_img.ndim == 3 and int(gpu_img.shape[0]) in (1, 3, 4) and int(gpu_img.shape[-1]) not in (1, 3, 4):
                    gpu_img = gpu_img.permute(1, 2, 0)
                if gpu_img.shape[-1] == 4:
                    gpu_img = gpu_img[..., :3]
                if gpu_img.shape[-1] == 1:
                    gpu_img = gpu_img.expand(-1, -1, 3)
                if gpu_img.dtype != torch.uint8:
                    gpu_img = gpu_img.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                gpu_img = gpu_img.contiguous()
                processed = apply_aspect_on_gpu(
                    gpu_img,
                    source_size=(w, h),
                    target_size=(tw, th),
                    fit_mode=self.fit_mode,
                    display_mode=self.display_mode,
                    input_size=self.input_size,
                )
                # download processed
                if self._host_rgb is None or tuple(self._host_rgb.shape) != tuple(processed.shape):
                    self._host_rgb = torch.empty(tuple(processed.shape), dtype=torch.uint8, device="cpu", pin_memory=True)
                self._host_rgb.copy_(processed, non_blocking=True)
                torch.cuda.current_stream(device=processed.device).synchronize()
                result = self._host_rgb.numpy()
                return result.copy() if self.copy_output else result
            # No aspect required (cover/stretch): plain CUDA -> RGB download.
            return self._cuda_to_rgb(image)
        # CPU path: use aspect module for consistent letterbox/crop/stretch (mirrors local viewer)
        rgb = runtime_sbs_to_rgb(image)
        h2, w2 = rgb.shape[0], rgb.shape[1]
        if not process_aspect:
            return rgb
        # Use same aspect logic as local viewer: input_size is tex_w,tex_h
        return apply_aspect_on_cpu(
            rgb,
            source_size=(w2, h2),
            target_size=(tw, th),
            fit_mode=self.fit_mode,
            display_mode=self.display_mode,
            input_size=self.input_size,
        )

    def _cuda_to_rgb(self, image) -> np.ndarray:
        import torch

        if image.ndim == 4:
            if int(image.shape[0]) != 1:
                raise ValueError(
                    f"expected one SBS frame, got shape {tuple(image.shape)!r}"
                )
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"unsupported SBS frame shape: {tuple(image.shape)!r}")

        if int(image.shape[0]) in (1, 3, 4):
            channels = int(image.shape[0])
            if channels == 1:
                image = image.expand(3, -1, -1)
            elif channels == 4:
                image = image[:3]
            image = image.permute(1, 2, 0)
        elif int(image.shape[-1]) in (1, 3, 4):
            channels = int(image.shape[-1])
            if channels == 1:
                image = image.expand(-1, -1, 3)
            elif channels == 4:
                image = image[..., :3]
        else:
            raise ValueError(
                f"unsupported SBS channel count for shape {tuple(image.shape)!r}"
            )

        if image.dtype != torch.uint8:
            image = image.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        image = image.contiguous()
        if (
            self._host_rgb is None
            or tuple(self._host_rgb.shape) != tuple(image.shape)
        ):
            self._host_rgb = torch.empty(
                tuple(image.shape),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
        self._host_rgb.copy_(image, non_blocking=True)
        torch.cuda.current_stream(device=image.device).synchronize()
        result = self._host_rgb.numpy()
        return result.copy() if self.copy_output else result


class DirectSbsOutputConsumer:
    """Consume only the newest runtime SBS frame and submit it to a stream sink."""

    def __init__(
        self,
        *,
        runtime_q,
        shutdown_event,
        output,
        source_stat_inc: Callable[..., None],
        show_fps_provider: Callable[[], bool] | None = None,
        on_sbs_fps: Callable[..., Any] | None = None,
        fps_report_interval: float = 5.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.runtime_q = runtime_q
        self.shutdown_event = shutdown_event
        self.output = output
        self.source_stat_inc = source_stat_inc
        self.show_fps_provider = show_fps_provider
        self.on_sbs_fps = on_sbs_fps
        self.fps_report_interval = max(0.1, float(fps_report_interval))
        self._clock = clock
        self._fps_started = self._clock()
        self._fps_sbs_frames = 0
        self._fps_submitted_frames = 0
        self._fps_convert_seconds = 0.0
        self._fps_submit_seconds = 0.0
        # Pass aspect config to converter for CPU fallback path
        self._frame_converter = RuntimeSbsRgbConverter(
            copy_output=not bool(getattr(output, "synchronous_submit", False)),
            display_mode=getattr(output, "display_mode", "Half-SBS"),
            fit_mode=getattr(output, "fit_mode", "contain"),
            input_size=getattr(output, "input_size", None),
        )

    def _apply_cuda_aspect(self, frame: Any) -> Any:
        """Pad a CUDA frame into the 16:9 transport canvas (contain only).

        The letterboxing runs entirely on the GPU (torch resize + blit into a
        device canvas), so CUDA paths stay zero-copy with respect to the host:
        the padded tensor is handed straight to the GPU encoder.
        """
        from streaming.aspect import apply_aspect_on_gpu, transport_canvas_size

        if frame.ndim == 4:
            h, w = int(frame.shape[-2]), int(frame.shape[-1])
        elif frame.ndim == 3 and int(frame.shape[0]) in (1, 3, 4) and int(frame.shape[-1]) not in (1, 3, 4):
            h, w = int(frame.shape[-2]), int(frame.shape[-1])
        else:
            h, w = int(frame.shape[0]), int(frame.shape[1])
        tw, th = transport_canvas_size(
            (w, h),
            "contain",
            input_size=getattr(self.output, "input_size", None),
            display_mode=getattr(self.output, "display_mode", "Half-SBS"),
        )
        if (tw, th) == (w, h):
            return frame
        return apply_aspect_on_gpu(
            frame,
            source_size=(w, h),
            target_size=(tw, th),
            fit_mode="contain",
            display_mode=getattr(self.output, "display_mode", "Half-SBS"),
            input_size=getattr(self.output, "input_size", None),
        )

    def _frame_letterbox_required(self, runtime_result: Any) -> bool:
        """Return whether this frame must be padded into the 16:9 canvas.

        Only the contain ("keep ratio complete") fit mode pads each eye to
        16:9, and only when the input aspect ratio is not already 16:9 (legacy
        ``fill_16_9`` semantics). Native GPU surface paths (Intel D3D11/oneVPL
        final-SBS and the deferred Vulkan compose) present the packed SBS at
        its native aspect and cannot letterbox, so the consumer must bypass
        them when this is required. The per-eye size is always taken from the
        actual frame (eyes, native surface, or packed sbs) so the gate can
        never disagree with the frame-derived transport canvas; the configured
        ``input_size`` is only a last resort.
        """
        if normalize_display_fit_mode(getattr(self.output, "fit_mode", "contain")) != "contain":
            return False
        from streaming.aspect import _input_eye_size, input_needs_16_9_canvas

        left_eye = getattr(runtime_result, "left_eye", None)
        if left_eye is not None and getattr(left_eye, "width", 0) and getattr(left_eye, "height", 0):
            return input_needs_16_9_canvas(
                (int(left_eye.width), int(left_eye.height))
            )
        native_surface = getattr(runtime_result, "native_final_sbs_surface", None)
        if native_surface is not None and getattr(native_surface, "width", 0) and getattr(native_surface, "height", 0):
            # The surface is the actual packed SBS this frame would send; derive
            # the per-eye size from it directly so the gate reflects the frame.
            sw, sh = int(native_surface.width), int(native_surface.height)
            eye_size = _input_eye_size(
                (sw, sh),
                getattr(self.output, "display_mode", "Half-SBS"),
                None,
            )
            return input_needs_16_9_canvas(eye_size)
        sbs = getattr(runtime_result, "sbs", None)
        if sbs is not None and getattr(sbs, "shape", None):
            from streaming.aspect import frame_hw

            h, w = frame_hw(sbs)
            if h > 0 and w > 0:
                eye_size = _input_eye_size(
                    (w, h),
                    getattr(self.output, "display_mode", "Half-SBS"),
                    None,
                )
                return input_needs_16_9_canvas(eye_size)
        input_size = getattr(self.output, "input_size", None)
        if input_size is None:
            return False
        try:
            eye_size = (int(input_size[0]), int(input_size[1]))
        except (TypeError, ValueError):
            return False
        if eye_size[0] <= 0 or eye_size[1] <= 0:
            return False
        return input_needs_16_9_canvas(eye_size)

    def _take_latest(self):
        try:
            item = self.runtime_q.get(timeout=0.05)
        except queue.Empty:
            return None
        if bool(getattr(self.runtime_q, "_d2s_ordered", False)):
            return item
        while True:
            try:
                item = self.runtime_q.get_nowait()
                self.source_stat_inc("runtime_output_overwrite")
            except queue.Empty:
                return item

    def _report_fps_if_due(self) -> None:
        now = self._clock()
        elapsed = now - self._fps_started
        if elapsed < self.fps_report_interval:
            return
        sbs_fps = self._fps_sbs_frames / elapsed
        submitted_fps = self._fps_submitted_frames / elapsed
        convert_ms = (
            self._fps_convert_seconds * 1000.0 / self._fps_sbs_frames
            if self._fps_sbs_frames
            else 0.0
        )
        submit_ms = (
            self._fps_submit_seconds * 1000.0 / self._fps_submitted_frames
            if self._fps_submitted_frames
            else 0.0
        )
        if self.on_sbs_fps is not None:
            self.on_sbs_fps(sbs_fps, frame_count=self._fps_sbs_frames)
        observe_calibration = getattr(self.output, "observe_calibration_window", None)
        if callable(observe_calibration):
            observe_calibration(
                sbs_fps=sbs_fps,
                submitted_fps=submitted_fps,
                convert_ms=convert_ms,
                submit_ms=submit_ms,
            )
        show_fps = (
            bool(self.show_fps_provider())
            if self.show_fps_provider is not None
            else False
        )
        if show_fps:
            network_bitrate = float(
                getattr(self.output, "current_network_bitrate_mbps", 0.0) or 0.0
            )
            print(
                f"[DirectSbsStream] SBS FPS: {sbs_fps:.1f} "
                f"network_bitrate={network_bitrate:.1f} Mbps "
                f"submitted={submitted_fps:.1f} "
                f"convert_ms={convert_ms:.1f} submit_ms={submit_ms:.1f}",
                flush=True,
            )
        self._fps_sbs_frames = 0
        self._fps_submitted_frames = 0
        self._fps_convert_seconds = 0.0
        self._fps_submit_seconds = 0.0
        self._fps_started = now

    def run(self) -> None:
        while not self.shutdown_event.is_set():
            item = self._take_latest()
            if item is None:
                continue
            try:
                runtime_result, _capture_timestamp = item
                self._fps_sbs_frames += 1
                prepare_calibration = getattr(
                    self.output, "prepare_calibration_source", None
                )
                if callable(prepare_calibration) and prepare_calibration(runtime_result):
                    self._fps_submitted_frames += 1
                    self.source_stat_inc("runtime_output_frames")
                    self.source_stat_inc("network_stream_frames")
                    self._report_fps_if_due()
                    continue
                should_submit = getattr(self.output, "should_submit_frame", None)
                if callable(should_submit) and not should_submit(self._clock()):
                    self._report_fps_if_due()
                    continue
                # Native GPU surface paths (Intel D3D11/oneVPL final-SBS and the
                # deferred Vulkan compose) present the packed SBS at its native
                # aspect and cannot letterbox into the 16:9 transport canvas.
                # When the input aspect ratio is not 16:9 (contain fit mode),
                # bypass them so this frame goes through the aspect-aware
                # CUDA/CPU paths below. The CUDA path keeps the letterboxing on
                # the GPU (zero host round trip) as the first priority.
                letterbox_required = self._frame_letterbox_required(runtime_result)
                if letterbox_required and not getattr(self, "_letterbox_notice", False):
                    self._letterbox_notice = True
                    print(
                        "[DirectSbsStream] Non-16:9 input: native GPU surface "
                        "paths bypassed; frames are padded into a 16:9 "
                        "transport canvas (legacy fill_16_9)",
                        flush=True,
                    )
                submit_vulkan_stereo = getattr(
                    self.output,
                    "submit_vulkan_stereo_frame",
                    None,
                )
                if letterbox_required:
                    submit_vulkan_stereo = None
                left_eye = getattr(runtime_result, "left_eye", None)
                right_eye = getattr(runtime_result, "right_eye", None)
                if callable(submit_vulkan_stereo) and getattr(
                    left_eye, "context", None
                ) is not None and getattr(right_eye, "context", None) is not None:
                    handled = submit_vulkan_stereo(runtime_result)
                    if handled is False:
                        self.source_stat_inc("runtime_output_vulkan_fallback")
                        self._report_fps_if_due()
                        continue
                    self._fps_submitted_frames += 1
                    self.source_stat_inc("runtime_output_frames")
                    self.source_stat_inc("network_stream_frames")
                    self._report_fps_if_due()
                    continue
                if callable(submit_vulkan_stereo) and getattr(
                    runtime_result, "vulkan_compute_request", None
                ) is not None:
                    handled = submit_vulkan_stereo(runtime_result)
                    if handled is False:
                        self.source_stat_inc("runtime_output_vulkan_fallback")
                        self._report_fps_if_due()
                        continue
                    self._fps_submitted_frames += 1
                    self.source_stat_inc("runtime_output_frames")
                    self.source_stat_inc("network_stream_frames")
                    self._report_fps_if_due()
                    continue
                native_surface = getattr(runtime_result, "native_final_sbs_surface", None)
                submit_native_surface = getattr(
                    self.output, "submit_native_d3d11_surface", None
                )
                if letterbox_required:
                    submit_native_surface = None
                if native_surface is not None and callable(submit_native_surface):
                    handled = submit_native_surface(native_surface)
                    if handled is False:
                        # The native surface may be video-only (for example Intel
                        # oneVPL). Preserve configured audio by sending this frame
                        # through the shared QSV/FFmpeg path instead of dropping it.
                        convert_started = self._clock()
                        frame = self._frame_converter.convert(runtime_result)
                        self._fps_convert_seconds += self._clock() - convert_started
                        submit_started = self._clock()
                        self.output.submit_frame(frame)
                        self._fps_submit_seconds += self._clock() - submit_started
                    self._fps_submitted_frames += 1
                    self.source_stat_inc("runtime_output_frames")
                    self.source_stat_inc("network_stream_frames")
                    self._report_fps_if_due()
                    continue
                submit_cuda_frame = getattr(self.output, "submit_cuda_frame", None)
                cuda_frame = getattr(runtime_result, "sbs", runtime_result)
                if callable(submit_cuda_frame) and bool(
                    getattr(cuda_frame, "is_cuda", False)
                ):
                    # GPU zerocopy paths skip RuntimeSbsRgbConverter, so apply the
                    # same aspect rule here: contain letterboxes into a 16:9
                    # transport canvas, cover/stretch pass the frame through.
                    if (
                        normalize_display_fit_mode(
                            getattr(self.output, "fit_mode", "contain")
                        )
                        == "contain"
                        # MJPEG applies aspect inside its encoder loop.
                        and not isinstance(self.output, MjpegDirectSbsOutput)
                    ):
                        # GPU letterboxing is mandatory for CUDA frames (never a
                        # silent CPU fallback); a failure must surface.
                        cuda_frame = self._apply_cuda_aspect(cuda_frame)
                    convert_started = self._clock()
                    submit_cuda_frame(cuda_frame)
                    self._fps_submit_seconds += self._clock() - convert_started
                else:
                    convert_started = self._clock()
                    frame = self._frame_converter.convert(runtime_result)
                    self._fps_convert_seconds += self._clock() - convert_started
                    submit_started = self._clock()
                    self.output.submit_frame(frame)
                    self._fps_submit_seconds += self._clock() - submit_started
                self._fps_submitted_frames += 1
                self.source_stat_inc("runtime_output_frames")
                self.source_stat_inc("network_stream_frames")
                self._report_fps_if_due()
            except Exception as exc:
                self.source_stat_inc(
                    "network_stream_errors",
                    last_error=f"{type(exc).__name__}: {exc}",
                )
                print(
                    f"[DirectSbsStream] output failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self.shutdown_event.set()
                return


class _PyNvDirectSbsOutputMixin:
    """Encode CUDA video with PyNvVideoCodec and mux optional PCM audio."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pynv_output: PyNvSrtVideoOutput | None = None
        self._pynv_encoder: Any | None = None
        self._fallback_output: FfmpegDirectSbsOutput | None = None
        self._pynv_gpu_id = 0
        self._native_nvenc_active = False

    def _start_pynv_audio(self) -> str | None:
        if not self.stereo_mix_device:
            return None
        if self.os_name != "Windows" or not self.stereo_mix_device.casefold().startswith(
            "soundcard:"
        ):
            raise RuntimeError(
                "PyNvVideoCodec audio mux currently requires Windows soundcard loopback"
            )
        device_name = self.stereo_mix_device.split(":", 1)[1].strip()
        self._soundcard_audio = SoundcardLoopbackSender(device_name or None)
        self._soundcard_audio.start()
        return self._soundcard_audio.ffmpeg_url

    def _pynv_output_args(self) -> list[str]:
        if self.protocol == "WEBRTC":
            return [
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                "-pkt_size",
                "1452",
                f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}"
                "?pkt_size=1452",
            ]
        return [
            "-f",
            "mpegts",
            "-mpegts_flags",
            "+resend_headers",
            f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}&pkt_size=1316",
        ]

    def _release_pynv_pipeline(self) -> None:
        if self._pynv_output is not None:
            try:
                self._pynv_output.close()
            except Exception:
                pass
            self._pynv_output = None
        close_encoder = getattr(self._pynv_encoder, "close", None)
        if callable(close_encoder):
            try:
                close_encoder()
            except Exception:
                pass
        self._pynv_encoder = None
        self._native_nvenc_active = False
        if self._soundcard_audio is not None:
            self._soundcard_audio.close()
            self._soundcard_audio = None

    def _start_ffmpeg(self, width: int, height: int) -> None:
        # GPU backends override this method. Calibration is an independent
        # deterministic FFmpeg pressure stream, so explicitly call the shared
        # base implementation instead of starting the vendor encoder.
        if self._calibration_controller is not None:
            return FfmpegDirectSbsOutput._start_ffmpeg(self, width, height)
        # PyNvVideoCodec exposes encoded Annex-B bytes without packet PTS/DTS.
        # FFmpeg cannot reliably repair that metadata during stream-copy RTSP
        # muxing. With audio enabled, use the shared Vulkan/FFmpeg A/V path,
        # which timestamps rawvideo and PCM in one muxer and keeps browser audio
        # reliable. Native NVENC remains available for video-only sessions.
        if self.stereo_mix_device and not self.stereo_mix_device.casefold().startswith(("no ", "none", "null")):
            self._fallback_output = FfmpegDirectSbsOutput(
                base_dir=self.base_dir,
                protocol=self.protocol,
                port=self.port,
                stream_key=self.stream_key,
                fps=self.fps,
                crf=self.crf,
                stereo_mix_device=self.stereo_mix_device,
                audio_delay=self.audio_delay,
                os_name=self.os_name,
                prefer_nvenc=self.prefer_nvenc,
                display_mode=self.display_mode,
            )
            self._fallback_output.server_process = self.server_process
            print(
                "[DirectSbsStream] NativeNVENC audio mux disabled: "
                "PyNv Annex-B packets have no PTS/DTS; using shared FFmpeg A/V path",
                flush=True,
            )
            return
        nvc = _load_pynvvideo_codec()
        if nvc is None:
            raise RuntimeError(f"PyNvVideoCodec unavailable: {_PYNVVIDEO_CODEC_ERROR}")
        audio_url = self._start_pynv_audio()
        codec = "hevc" if self.use_hevc else "h264"
        self._pynv_encoder = PyNvVideoCodecEncoder(
            nvc,
            width,
            height,
            hevc=self.use_hevc,
            fps=self.fps,
            bitrate=max(
                1,
                int(
                    (self._dynamic_stream_rate_budget(width, height) or (10,))[0]
                    * 1_000_000
                ),
            ),
            gpu_id=self._pynv_gpu_id,
        )
        self._pynv_output = PyNvSrtVideoOutput(
            self._pynv_encoder,
            str(self.ffmpeg_path),
            codec=codec,
            fps=self.fps,
            audio_url=audio_url,
            audio_delay=self.audio_delay,
            audio_codec="libopus" if self.protocol == "WEBRTC" else "aac",
            output_args=self._pynv_output_args(),
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if self.os_name == "Windows"
                else 0
            ),
        )
        time.sleep(0.05)
        if self._pynv_output.process.poll() is not None:
            raise RuntimeError(
                "PyNvVideoCodec FFmpeg muxer exited during startup with code "
                f"{self._pynv_output.process.returncode}"
            )
        self._frame_size = (width, height)
        audio_codec_label = "Opus" if self.protocol == "WEBRTC" else "AAC"
        audio_label = f" + SoundCard/{audio_codec_label}" if audio_url else ""
        print(
            f"[DirectSbsStream] PyNvVideoCodec {codec} GPU path active"
            f"{audio_label}: {width}x{height}@{self.fps} gpu_id={self._pynv_gpu_id} "
            "cuda_stream_handoff=synchronized",
            flush=True,
        )

    def _start_cudaarray_encoder(
        self, width: int, height: int, cuda_array: int
    ) -> None:
        """Start NativeNVENC and publish timestamped H264/Opus over RTSP/RTP."""
        if self.use_hevc:
            raise RuntimeError(
                "NativeNVENC RTSP/RTP baseline currently supports H.264 only; "
                "HEVC requires the H.265 RTP packetizer"
            )
        self._start_pynv_audio()
        bitrate = max(
            1,
            int(
                (self._dynamic_stream_rate_budget(width, height) or (10,))[0]
                * 1_000_000
            ),
        )
        self._native_nvenc_active = True
        self._pynv_encoder = NvencCudaArrayEncoder(
            width,
            height,
            hevc=self.use_hevc,
            fps=self.fps,
            bitrate=bitrate,
            cuda_array=cuda_array,
        )
        self._pynv_output = NativeRtspAvOutput(
            self._pynv_encoder,
            host="127.0.0.1",
            port=self.publish_rtsp_port,
            stream_key=self.stream_key,
            fps=self.fps,
            audio_sender=self._soundcard_audio,
        )
        self._pynv_output.start()
        self._frame_size = (width, height)

    def _fallback_to_ffmpeg(self, frame: Any, reason: Exception) -> None:
        if self._native_nvenc_active:
            print(
                f"[DirectSbsStream] NativeNVENC runtime failure: {reason}; "
                "FFmpeg fallback is disabled for the NativeNVENC path",
                flush=True,
            )
            raise RuntimeError("NativeNVENC path failed without FFmpeg fallback") from reason
        print(
            f"[DirectSbsStream] PyNvVideoCodec runtime failure: {reason}; "
            "falling back to FFmpeg video/audio encoding",
            flush=True,
        )
        self._release_pynv_pipeline()
        self._fallback_output = FfmpegDirectSbsOutput(
            base_dir=self.base_dir,
            protocol=self.protocol,
            port=self.port,
            stream_key=self.stream_key,
            fps=self.fps,
            crf=self.crf,
            stereo_mix_device=self.stereo_mix_device,
            audio_delay=self.audio_delay,
            os_name=self.os_name,
            prefer_nvenc=self.prefer_nvenc,
            display_mode=self.display_mode,
        )
        self._fallback_output.server_process = self.server_process
        self._fallback_output.submit_frame(runtime_sbs_to_rgb(frame))

    def submit_cuda_frame(self, frame: Any) -> None:
        if self._fallback_output is not None:
            self._fallback_output.submit_frame(runtime_sbs_to_rgb(frame))
            return
        try:
            if self._pynv_output is None:
                device = getattr(frame, "device", None)
                device_index = getattr(device, "index", None)
                if device_index is not None:
                    self._pynv_gpu_id = max(0, int(device_index))
                height, width = int(frame.shape[-2]), int(frame.shape[-1])
                if int(frame.shape[0]) not in (1, 3, 4):
                    height, width = int(frame.shape[0]), int(frame.shape[1])
                self._start_ffmpeg(width, height)
            if self._fallback_output is not None:
                self._fallback_output.submit_frame(runtime_sbs_to_rgb(frame))
                return
            assert self._pynv_output is not None
            self._pynv_output.submit_cuda_frame(frame)
        except Exception as exc:
            self._fallback_to_ffmpeg(frame, exc)

    def close(self) -> None:
        self._release_pynv_pipeline()
        if self._fallback_output is not None:
            self._fallback_output.close()
            self._fallback_output = None
        super().close()


class MjpegDirectSbsOutput:
    """
    MJPEG streaming output with probe-first rate selection and GPU zerocopy aspect processing.
    Mirrors FfmpegDirectSbsOutput rate calibration logic for consistent behavior.
    """
    synchronous_submit = False

    def __init__(
        self,
        *,
        port: int,
        fps: int,
        quality: int,
        display_mode: str = "Half-SBS",
        fit_mode: str = "contain",
        input_size: Tuple[int, int] | None = None,
        on_stream_fps_selected: Callable[[int], Any] | None = None,
    ) -> None:
        self.port = max(1, int(port))
        self.requested_fps = max(1, int(fps))
        self.fps = self.requested_fps
        self.quality = max(1, min(100, int(quality)))
        self.display_mode = str(display_mode or "Half-SBS").strip()
        self.fit_mode = str(fit_mode or "contain").strip()
        self.input_size = input_size
        self._on_stream_fps_selected = on_stream_fps_selected

        profile = EncoderProfile(
            codec="mjpeg",
            quality=self.quality,
            target_fps=self.requested_fps,
            pixel_format="rgb",
        )
        self.streamer = MJPEGStreamer(
            port=self.port,
            profile=profile,
            display_mode=self.display_mode,
            fit_mode=self.fit_mode,
            input_size=self.input_size,
        )

    def start(self) -> None:
        # Kill orphan MediaMTX / streaming ports from previous run (esp. when switching modes)
        try:
            _kill_orphan_mediamtx_processes(ports=[self.port, 9998, 8000, 8001, 8189])
        except Exception:
            pass
        self.streamer.start()
        print("[DirectSbsStream] MJPEG consumes packed SBS frames directly", flush=True)

    def submit_frame(self, frame: np.ndarray) -> None:
        self.streamer.set_frame(frame)

    def submit_cuda_frame(self, frame: Any) -> None:
        """
        Submit CUDA frame for zerocopy processing (aspect/resize on GPU, JPEG on CPU).
        frame: torch.Tensor on CUDA
        """
        import torch
        if not (hasattr(frame, "is_cuda") and frame.is_cuda):
            # Fallback to CPU path
            self.submit_frame(runtime_sbs_to_rgb(frame))
            return
        # Create CUDA event for synchronization
        cuda_event = torch.cuda.Event()
        cuda_event.record(torch.cuda.current_stream(frame.device))
        self.streamer.set_cuda_frame(frame, cuda_event)

    @property
    def current_network_bitrate_mbps(self) -> float:
        """MJPEG has no MediaMTX bitrate tracking."""
        return 0.0

    def should_submit_frame(self, now: float | None = None) -> bool:
        """MJPEG streams every runtime frame immediately.

        There is no network-rate probe for MJPEG: dropping frames for a
        multi-second calibration window froze the stream at startup and the
        0.9x sustainable-rate cap needlessly throttled it below the runtime
        rate. The encoder loop keeps only the newest frame and _generate paces
        the HTTP output at ``delay``, so rate control is already handled.
        """
        return True

    def observe_calibration_window(
        self,
        *,
        sbs_fps: float,
        submitted_fps: float,
        convert_ms: float,
        submit_ms: float,
    ) -> None:
        # MJPEG has no calibration controller, but keep interface for consumer
        pass

    def close(self) -> None:
        self.streamer.stop()


class FfmpegDirectSbsOutput:
    """Publish RGB SBS frames to MediaMTX through FFmpeg rawvideo stdin."""

    synchronous_submit = True

    def __init__(
        self,
        *,
        base_dir: str | Path,
        protocol: str,
        port: int,
        stream_key: str,
        fps: int,
        crf: int,
        stereo_mix_device: str | None = None,
        audio_delay: float = -0.1,
        os_name: str | None = None,
        prefer_nvenc: bool = False,
        display_mode: str = "Half-SBS",
        fit_mode: str = "contain",
        input_size: Tuple[int, int] | None = None,
        target_bitrate_mbps: int = 0,
        peak_bitrate_mbps: int = 0,
        auto_calibration: bool = False,
        calibration_port: int | None = None,
        on_calibration_fps: Callable[[int], Any] | None = None,
        calibration_fingerprint: dict[str, str] | None = None,
        on_stream_fps_selected: Callable[[int], Any] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.protocol = str(protocol or "RTMP").strip().upper()
        self.port = max(1, int(port))
        self.stream_key = str(stream_key or "live").strip() or "live"
        self.fps = max(1, int(fps))
        self.requested_fps = self.fps
        self.crf = max(0, min(51, int(crf)))
        self.stereo_mix_device = str(stereo_mix_device or "").strip()
        self.audio_delay = float(audio_delay)
        self._soundcard_audio: SoundcardLoopbackSender | None = None
        self.os_name = str(os_name or platform.system())
        self.prefer_nvenc = bool(prefer_nvenc)
        self.display_mode = str(display_mode or "Half-SBS").strip()
        self.fit_mode = str(fit_mode or "contain").strip()
        self.input_size = input_size
        if self.input_size is not None:
            try:
                self.input_size = (int(self.input_size[0]), int(self.input_size[1]))
            except Exception:
                self.input_size = None
        self.use_hevc = self.display_mode.casefold() == "full-sbs"
        self.target_bitrate_mbps = max(0, int(target_bitrate_mbps))
        self.peak_bitrate_mbps = max(0, int(peak_bitrate_mbps))
        self.auto_calibration = bool(auto_calibration and self.protocol == "WEBRTC")
        self.calibration_port = int(calibration_port or min(65535, self.port + 1))
        self._on_calibration_fps = on_calibration_fps
        self._on_stream_fps_selected = on_stream_fps_selected
        self._calibration_controller: StreamCalibrationController | None = None
        self.video_encoder = "libx265" if self.use_hevc else "libx264"
        self._active_rate_budget: tuple[int, int, int] | None = None
        self._encoder_selected = False
        self._qsv_surface_mode = "host_upload"
        runtime_root = Path(
            os.environ.get(
                "D2S_STREAMING_RUNTIME_DIR",
                self.base_dir / "streaming" / "rtmp",
            )
        )
        ffmpeg_name = "ffmpeg.exe" if self.os_name == "Windows" else "ffmpeg"
        mediamtx_name = "mediamtx.exe" if self.os_name == "Windows" else "mediamtx"
        if (runtime_root / "runtime-manifest.json").is_file():
            ensure_runtime(runtime_root)
        self.ffmpeg_path = self._find_executable(
            "D2S_FFMPEG_PATH",
            runtime_root / "ffmpeg" / "bin" / ffmpeg_name,
            "ffmpeg",
        )
        self.mediamtx_path = self._find_executable(
            "D2S_MEDIAMTX_PATH",
            runtime_root / "mediamtx" / mediamtx_name,
            "mediamtx",
        )
        self.mediamtx_config = Path(
            os.environ.get(
                "D2S_MEDIAMTX_CONFIG",
                runtime_root / "mediamtx.yml",
            )
        )
        if not self.mediamtx_config.is_file():
            raise FileNotFoundError(f"MediaMTX config not found: {self.mediamtx_config}")
        if sys.platform == "darwin":
            # MediaMTX cannot apply udpReadBufferSize on macOS ("read buffer
            # size is unimplemented on the current operating system") and
            # aborts at startup, killing every listener. Drop it to the OS
            # default in a runtime-generated copy; Windows/Linux keep the
            # enlarged buffer.
            self.mediamtx_config = _darwin_safe_mediamtx_config(self.mediamtx_config)
        self.server_process: subprocess.Popen | None = None
        self.ffmpeg_process: subprocess.Popen | None = None
        self._ffmpeg_log_thread: threading.Thread | None = None
        self._ffmpeg_stderr_tail: list[str] = []
        self._ffmpeg_bitrate_mbps = 0.0
        self._mediamtx_inbound_bitrate_mbps = 0.0
        self._mediamtx_metrics_stop = threading.Event()
        self._mediamtx_metrics_thread: threading.Thread | None = None
        self._mediamtx_metrics_address = "127.0.0.1:9998"
        self._server_log_thread: threading.Thread | None = None
        self._frame_size: tuple[int, int] | None = None
        self._rate_probe_started: float | None = None
        self._rate_window_started: float | None = None
        self._rate_window_frames = 0
        self._rate_window_fps: list[float] = []
        self._rate_probe_min_seconds = 5.0
        self._rate_probe_max_seconds = 15.0
        self._stream_rate_calibrated = False
        self._next_submit_at = 0.0
        self._pending_audio_delay: float | None = None
        self._audio_delay_lock = threading.Lock()
        self._packet_loss_warning_emitted = False
        self._darwin_audio_device: str | None = None
        self._darwin_audio_probe_started = False
        self._audio_startup_retried = False
        if self.auto_calibration:
            logs_dir = self.base_dir / "logs"
            self._calibration_controller = StreamCalibrationController(
                bind_port=self.calibration_port,
                stream_port=self.port,
                stream_key=self.stream_key,
                maximum_fps=self.requested_fps,
                state_path=logs_dir / "stream_calibration_state.json",
                profile_path=logs_dir / "stream_calibration_profile.json",
                hevc=self.use_hevc,
                fingerprint=calibration_fingerprint,
            )
            self._stream_rate_calibrated = True

    @property
    def current_network_bitrate_mbps(self) -> float:
        """Return the latest measured bitrate at the network publish boundary."""
        return float(
            self._mediamtx_inbound_bitrate_mbps
            or self._ffmpeg_bitrate_mbps
            or 0.0
        )

    def prepare_calibration_source(self, runtime_result: Any) -> bool:
        """Start or retune the independent probe without converting RGB frames."""
        if self._calibration_controller is None:
            return False
        display_size = getattr(runtime_result, "output_display_size", None)
        if not display_size:
            frame = getattr(runtime_result, "sbs", runtime_result)
            shape = tuple(int(value) for value in getattr(frame, "shape", ()))
            if len(shape) == 4:
                display_size = (shape[-1], shape[-2])
            elif len(shape) == 3 and shape[0] in {1, 3, 4}:
                display_size = (shape[-1], shape[-2])
            elif len(shape) == 3:
                display_size = (shape[1], shape[0])
            else:
                raise ValueError("Unable to determine calibration stream resolution")
        output_width, output_height = (int(display_size[0]), int(display_size[1]))
        input_width = output_width
        if self.display_mode.casefold() == "full-sbs":
            input_width = max(1, output_width // 2)
        self._calibration_controller.configure_input_resolution(
            input_width, output_height
        )
        self._apply_pending_calibration_tier()
        size = (output_width, output_height)
        if self.ffmpeg_process is None:
            self._start_ffmpeg(*size)
        elif self._frame_size != size:
            raise RuntimeError(
                f"Calibration stream size changed from {self._frame_size} to {size}"
            )
        return True

    @staticmethod
    def _find_executable(env_name: str, bundled: Path, command: str) -> Path:
        configured = os.environ.get(env_name)
        candidates = [Path(configured)] if configured else []
        candidates.append(bundled)
        discovered = shutil.which(command)
        if discovered:
            candidates.append(Path(discovered))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            f"{command} not found; set {env_name} or install it under {bundled.parent}"
        )

    def _server_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # Keep MediaMTX path byte counters enabled for normal streaming too;
        # the GUI FPS overlay uses them as the network bitrate source after
        # automatic calibration has finished.
        env["MTX_METRICS"] = "yes"
        env["MTX_METRICSADDRESS"] = self._mediamtx_metrics_address
        if self.protocol == "RTMP":
            env["MTX_RTMPADDRESS"] = f":{self.port}"
        elif self.protocol == "RTSP":
            env["MTX_RTSPADDRESS"] = f":{self.port}"
        elif self.protocol in {"HLS", "HLS M3U8"}:
            env["MTX_HLSADDRESS"] = f":{self.port}"
        elif self.protocol == "WEBRTC":
            env["MTX_WEBRTCADDRESS"] = f":{self.port}"
        return env

    @staticmethod
    def _parse_mediamtx_path_inbound_bytes(metrics: str, path: str) -> int | None:
        for line in str(metrics or "").splitlines():
            if not line.startswith(("paths_inbound_bytes{", "paths_bytes_received{")):
                continue
            labels, separator, value = line.partition("}")
            if not separator or f'name="{path}"' not in labels:
                continue
            try:
                return int(float(value.strip()))
            except ValueError:
                return None
        return None

    def _read_mediamtx_path_inbound_bytes(self) -> int | None:
        try:
            with urlopen(
                f"http://{self._mediamtx_metrics_address}/metrics"
                f"?type=paths&path={quote(self.stream_key)}",
                timeout=0.5,
            ) as response:
                payload = response.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        return self._parse_mediamtx_path_inbound_bytes(payload, self.stream_key)

    def _sample_mediamtx_metrics(self) -> None:
        previous_bytes: int | None = None
        previous_time: float | None = None
        while not self._mediamtx_metrics_stop.wait(1.0):
            current_bytes = self._read_mediamtx_path_inbound_bytes()
            current_time = time.monotonic()
            if current_bytes is None:
                continue
            if (
                previous_bytes is not None
                and previous_time is not None
                and current_bytes >= previous_bytes
            ):
                elapsed = current_time - previous_time
                if elapsed > 0:
                    self._mediamtx_inbound_bitrate_mbps = (
                        (current_bytes - previous_bytes) * 8.0 / elapsed / 1_000_000.0
                    )
            previous_bytes = current_bytes
            previous_time = current_time

    @property
    def publish_rtsp_port(self) -> int:
        return self.port if self.protocol == "RTSP" else 8554

    def _probe_encoder(self, encoder: str, width: int, height: int) -> bool:
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if self.os_name == "Windows"
            else 0
        )
        command = [
            str(self.ffmpeg_path), "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={int(width)}x{int(height)}:r=1",
            "-frames:v", "1", "-an",
        ]
        if encoder.endswith("_vaapi"):
            vaapi_device = os.environ.get("D2S_VAAPI_DEVICE", "/dev/dri/renderD128")
            command[1:1] = ["-vaapi_device", vaapi_device]
            command.extend(["-vf", "format=nv12,hwupload"])
        command.extend([
            "-c:v",
            encoder,
            "-pix_fmt",
            "nv12" if encoder.endswith("_vaapi") else "yuv420p",
            "-f",
            "null",
            "-",
        ])
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=8.0,
                creationflags=creationflags,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(
                f"[DirectSbsStream] {encoder} probe failed for {int(width)}x{int(height)}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return False
        if result.returncode == 0:
            return True
        stderr = result.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        detail_lines = [line.strip() for line in str(stderr or "").splitlines() if line.strip()]
        detail = detail_lines[-1] if detail_lines else f"FFmpeg exited with code {result.returncode}"
        if encoder.endswith("_nvenc"):
            print(
                f"[DirectSbsStream] NVENC probe failed for {int(width)}x{int(height)}: {detail}",
                flush=True,
            )
        else:
            print(f"[DirectSbsStream] {encoder} unavailable: {detail}", flush=True)
        return False

    def _probe_nvenc(self, width: int, height: int) -> bool:
        if self.os_name == "Darwin":
            return False
        return self._probe_encoder(
            "hevc_nvenc" if self.use_hevc else "h264_nvenc", width, height
        )

    def _encoder_candidates(self) -> list[tuple[str, str]]:
        codec = "hevc" if self.use_hevc else "h264"
        candidates: list[tuple[str, str]] = []
        if self.os_name == "Darwin":
            candidates.append((f"{codec}_videotoolbox", "Apple VideoToolbox"))
        else:
            if self.os_name == "Windows":
                candidates.append((f"{codec}_nvenc", "NVIDIA NVENC"))
                candidates.extend([
                    (f"{codec}_qsv", "Intel Quick Sync"),
                    (f"{codec}_amf", "AMD AMF"),
                ])
            elif self.os_name == "Linux":
                candidates.extend([
                    (f"{codec}_qsv", "Intel Quick Sync"),
                    (f"{codec}_vaapi", "VAAPI"),
                ])
        software_encoder = "libx265" if self.use_hevc else "libx264"
        candidates.append((software_encoder, "software"))
        return candidates

    def _select_video_encoder(self, width: int, height: int) -> str:
        codec_label = "H.265/HEVC" if self.use_hevc else "H.264"
        candidates = self._encoder_candidates()
        for encoder, label in candidates[:-1]:
            supported = (
                self._probe_nvenc(width, height)
                if encoder.endswith("_nvenc")
                else self._probe_encoder(encoder, width, height)
            )
            if supported:
                self._encoder_selection_reason = label
                print(
                    f"[DirectSbsStream] {label} {codec_label} encoder active: {encoder}",
                    flush=True,
                )
                return encoder
        software_encoder, _ = candidates[-1]
        self._encoder_selection_reason = "software fallback"
        print(
            f"[DirectSbsStream] {codec_label} hardware encoders unavailable; "
            f"falling back to {software_encoder}",
            flush=True,
        )
        return software_encoder

    def _dynamic_stream_rate_budget(
        self, width: int, height: int
    ) -> tuple[int, int, int] | None:
        """Return wireless-friendly target, peak and VBV rates in Mbps."""
        if self.protocol not in {"HLS", "HLS M3U8", "RTMP", "WEBRTC"}:
            return None
        configured_target = int(getattr(self, "target_bitrate_mbps", 0) or 0)
        if configured_target > 0:
            configured_peak = int(getattr(self, "peak_bitrate_mbps", 0) or 0)
            peak = max(configured_target, configured_peak)
            return configured_target, peak, peak
        pixels_per_second = max(1, int(width)) * max(1, int(height)) * self.fps
        bits_per_pixel = 0.075 if self.use_hevc else 0.12
        quality_factor = max(0.5, min(2.0, 2.0 ** ((20 - self.crf) / 12.0)))
        target_limit = 100 if self.use_hevc else 80
        peak_limit = 120 if self.use_hevc else 100
        target_mbps = round(
            pixels_per_second * bits_per_pixel * quality_factor / 1_000_000
        )
        target_mbps = max(4, min(target_limit, target_mbps))
        peak_mbps = max(
            target_mbps,
            min(peak_limit, int(math.ceil(target_mbps * 1.15))),
        )
        return target_mbps, peak_mbps, peak_mbps

    @staticmethod
    def _select_sustainable_stream_fps(
        measured_fps: float, maximum_fps: int
    ) -> int:
        maximum = max(1, int(maximum_fps))
        measured = max(1.0, float(measured_fps))
        if measured >= float(maximum):
            return maximum
        safe_limit = min(float(maximum), measured * 0.90)
        for candidate in (60, 50, 48, 40, 30, 25, 24, 20, 15, 12, 10):
            if candidate <= maximum and candidate <= safe_limit:
                return candidate
        return max(5, min(maximum, int(safe_limit)))

    @staticmethod
    def _stable_rate_sample(window_fps: list[float]) -> float | None:
        recent = window_fps[-5:]
        if len(recent) < 5:
            return None
        median_fps = float(statistics.median(recent))
        if statistics.pstdev(recent) > max(1.0, median_fps * 0.06):
            return None
        ordered = sorted(recent)
        return float(ordered[max(0, int((len(ordered) - 1) * 0.20))])

    @staticmethod
    def _fallback_rate_sample(window_fps: list[float]) -> float:
        ordered = sorted(window_fps[-10:])
        if not ordered:
            return 1.0
        return float(ordered[max(0, int((len(ordered) - 1) * 0.20))])

    def should_submit_frame(self, now: float | None = None) -> bool:
        self._apply_pending_calibration_tier()
        timestamp = time.perf_counter() if now is None else float(now)
        if not self._stream_rate_calibrated:
            if self._rate_probe_started is None:
                self._rate_probe_started = timestamp
                self._rate_window_started = timestamp
                self._rate_window_frames = 1
                return False

            self._rate_window_frames += 1
            window_elapsed = timestamp - float(self._rate_window_started)
            if window_elapsed >= 1.0:
                self._rate_window_fps.append(
                    self._rate_window_frames / window_elapsed
                )
                self._rate_window_started = timestamp
                self._rate_window_frames = 0

            elapsed = timestamp - self._rate_probe_started
            stable_fps = self._stable_rate_sample(self._rate_window_fps)
            if elapsed < self._rate_probe_min_seconds or (
                stable_fps is None and elapsed < self._rate_probe_max_seconds
            ):
                return False
            measured_fps = (
                stable_fps
                if stable_fps is not None
                else self._fallback_rate_sample(self._rate_window_fps)
            )
            self.fps = self._select_sustainable_stream_fps(
                measured_fps, self.requested_fps
            )
            self._stream_rate_calibrated = True
            self._next_submit_at = timestamp + 1.0 / float(self.fps)
            if self._on_stream_fps_selected is not None:
                self._on_stream_fps_selected(self.fps)
            print(
                f"[DirectSbsStream] Stable stream rate selected: "
                f"measured={measured_fps:.1f} target={self.fps} FPS "
                f"windows={len(self._rate_window_fps)}",
                flush=True,
            )
            return True

        interval = 1.0 / float(self.fps)
        if timestamp + 1e-9 < self._next_submit_at:
            return False
        if timestamp - self._next_submit_at > interval:
            self._next_submit_at = timestamp + interval
        else:
            self._next_submit_at += interval
        return True

    def _apply_pending_calibration_tier(self) -> None:
        controller = getattr(self, "_calibration_controller", None)
        if controller is None:
            return
        tier = controller.take_pending_tier()
        if tier is None:
            return
        changed = (
            self.fps != tier.fps
            or self.target_bitrate_mbps != tier.target_mbps
            or self.peak_bitrate_mbps != tier.peak_mbps
        )
        self.fps = tier.fps
        self.target_bitrate_mbps = tier.target_mbps
        self.peak_bitrate_mbps = tier.peak_mbps
        self._next_submit_at = 0.0
        if callable(self._on_calibration_fps):
            self._on_calibration_fps(tier.fps)
        if changed and self.ffmpeg_process is not None:
            print(
                f"[StreamCalibration] Testing {tier.fps} FPS "
                f"target={tier.target_mbps}M peak={tier.peak_mbps}M",
                flush=True,
            )
            self._stop_process(self.ffmpeg_process)
            self.ffmpeg_process = None
            self._frame_size = None

    def observe_calibration_window(
        self,
        *,
        sbs_fps: float,
        submitted_fps: float,
        convert_ms: float,
        submit_ms: float,
    ) -> None:
        if getattr(self, "_calibration_controller", None) is None:
            return
        measured_bitrate = float(
            self._mediamtx_inbound_bitrate_mbps
            or self._ffmpeg_bitrate_mbps
            or 0.0
        )
        self._calibration_controller.observe_sender(
            {
                "sbs_fps": round(float(sbs_fps), 3),
                "submitted_fps": (
                    30.0
                    if self._calibration_controller is not None
                    else round(float(submitted_fps), 3)
                ),
                "convert_ms": round(float(convert_ms), 3),
                "submit_ms": round(float(submit_ms), 3),
                "encoded_bitrate_mbps": round(measured_bitrate, 3),
            }
        )

    @staticmethod
    def _looks_like_packet_loss(message: str) -> bool:
        normalized = str(message or "").casefold()
        indicators = (
            "packet loss",
            "packet lost",
            "packet missed",
            "missing packet",
            "dropped packet",
            "rtp packet gap",
            "buffer underflow",
        )
        return any(indicator in normalized for indicator in indicators)

    def _drain_mediamtx_output(self, process: subprocess.Popen) -> None:
        stream = process.stdout
        if stream is None:
            return
        for line in stream:
            message = str(line).rstrip("\r\n")
            if not message:
                continue
            print(f"[MediaMTX] {message}", flush=True)
            if self._looks_like_packet_loss(message) and not self._packet_loss_warning_emitted:
                self._packet_loss_warning_emitted = True
                print(
                    "[DirectSbsStream] WARNING: MediaMTX reported possible UDP packet loss; "
                    "consider increasing udpReadBufferSize (currently 0) and check network/CPU load.",
                    flush=True,
                )

    def start(self) -> None:
        # Kill orphan MediaMTX from previous run (port already in use -> ERR listen udp :8000)
        try:
            _kill_orphan_mediamtx_processes(ports=[self.port, 9998, 8000, 8001, 8189, 8554, 1935, 8888, 8889, 8890])
        except Exception:
            pass
        if self.protocol != "WEBRTC":
            print(
                f"[DirectSbsStream] WARNING: {self.protocol} selected; "
                "WebRTC is recommended for lower-latency browser streaming.",
                flush=True,
            )
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if self.os_name == "Windows"
            else 0
        )
        self.server_process = subprocess.Popen(
            [str(self.mediamtx_path), str(self.mediamtx_config)],
            cwd=str(self.mediamtx_config.parent),
            env=self._server_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )
        time.sleep(0.25)
        if self.server_process.poll() is not None:
            startup_output = ""
            try:
                startup_output, _ = self.server_process.communicate(timeout=1.0)
            except (OSError, subprocess.SubprocessError):
                pass
            output_lines = [
                line.strip() for line in startup_output.splitlines() if line.strip()
            ]
            detail = next(
                (line for line in reversed(output_lines) if " ERR " in line),
                output_lines[-1] if output_lines else "unknown startup error",
            )
            raise RuntimeError(f"MediaMTX exited during startup: {detail}")
        self._server_log_thread = threading.Thread(
            target=self._drain_mediamtx_output,
            args=(self.server_process,),
            name="MediaMTXLog",
            daemon=True,
        )
        self._server_log_thread.start()
        self._mediamtx_metrics_stop.clear()
        self._mediamtx_metrics_thread = threading.Thread(
            target=self._sample_mediamtx_metrics,
            name="MediaMTXMetrics",
            daemon=True,
        )
        self._mediamtx_metrics_thread.start()
        if self._calibration_controller is not None:
            self._calibration_controller.start()
        print(
            f"[DirectSbsStream] MediaMTX started for {self.protocol} on port {self.port}",
            flush=True,
        )

    def _audio_input_args(self) -> list[str]:
        device = self.stereo_mix_device
        if self.os_name == "Darwin":
            # No configured device (or a "no device found" placeholder):
            # auto-pick a loopback capture device so the stream always
            # carries sound on macOS.
            if not device or device.lower().startswith(("no ", "none", "null")):
                device = _auto_select_darwin_audio(self.ffmpeg_path)
        if not device or device.lower().startswith(("no ", "none", "null")):
            return []
        if self.os_name == "Windows":
            if device.casefold().startswith("soundcard:"):
                # Fix broken audio: revert to old stable dshow path (gfxcapture+ffmpeg handles loopback directly)
                # Old main.py used: -f dshow -rtbufsize 256M -i audio={device} with -fflags nobuffer
                # Python wasapi UDP loopback (s16le udp://...) causes fragmentation/discontinuity -> broken sound
                device_name = device.split(":", 1)[1].strip()
                if not device_name:
                    # The GUI stores an empty Stereo Mix device as the bare
                    # "soundcard:" prefix. Auto-pick a working dshow capture
                    # device (loopback preferred) so FFmpeg never receives an
                    # empty "-i audio=" name, which fails with "Error opening
                    # input: I/O error" and aborts the whole stream.
                    device_name = _auto_select_windows_audio(self.ffmpeg_path)
                    if not device_name:
                        return []
                return [
                    "-itsoffset",
                    str(self.audio_delay),
                    "-f",
                    "dshow",
                    "-rtbufsize",
                    "256M",
                    "-i",
                    f"audio={device_name}",
                ]
            if device.casefold().startswith("wasapi:"):
                wasapi_name = device.split(":", 1)[1].strip()
                if not wasapi_name:
                    # Same empty-device guard as the soundcard branch: skip
                    # audio instead of passing an unusable "-i " argument.
                    return []
                return [
                    "-itsoffset",
                    str(self.audio_delay),
                    "-f",
                    "wasapi",
                    "-i",
                    wasapi_name,
                ]
            return [
                "-itsoffset",
                str(self.audio_delay),
                "-f",
                "dshow",
                "-i",
                f"audio={device}",
            ]
        if self.os_name == "Linux":
            return [
                "-itsoffset",
                str(self.audio_delay),
                "-f",
                "pulse",
                "-i",
                device,
            ]
        if self.os_name == "Darwin":
            audio_device = device
            # Device labels from the GUI carry a backend prefix
            # ("soundcard:BlackHole 2ch", "wasapi:..."); AVFoundation matches
            # on the bare device name and rejects ":soundcard:BlackHole 2ch".
            for prefix in ("soundcard:", "wasapi:"):
                if audio_device.casefold().startswith(prefix):
                    audio_device = audio_device.split(":", 1)[1].strip()
                    break
            auto_selected = False
            if not audio_device or audio_device == ":":
                # No usable device (empty, bare "soundcard:" prefix, or a lone
                # ":"): skip audio entirely instead of handing FFmpeg "-i :"
                # which fails with "Error opening input file :." and kills the
                # stream.
                auto = _auto_select_darwin_audio(self.ffmpeg_path)
                if not auto:
                    return []
                auto_selected = True
                audio_device = auto
            if not audio_device.isdigit():
                # A configured device NAME (v2.5.0 settings parity) is
                # resolved to its AVFoundation index so the persisted "Stereo
                # Mix" device is captured even if the GUI list changes; a
                # stale/unplugged name falls back to auto-selection instead
                # of failing FFmpeg at startup.
                resolved_index = None
                for index, name in _list_darwin_audio_devices(self.ffmpeg_path):
                    if name.casefold() == audio_device.casefold():
                        resolved_index = str(index)
                        break
                if resolved_index is None:
                    print(
                        "[DirectSbsStream] WARNING: configured macOS Stereo Mix "
                        f"device {audio_device!r} is not available; auto-selecting",
                        flush=True,
                    )
                    resolved_index = _auto_select_darwin_audio(self.ffmpeg_path)
                    auto_selected = True
                if not resolved_index:
                    return []
                audio_device = resolved_index
            audio_device = f":{audio_device}"
            self._darwin_audio_device = audio_device
            # v2.5.0 macOS parity: the AVFoundation audio input carries the
            # same 256 MB ring buffer (rtbufsize) as the reference build.
            args = [
                "-itsoffset",
                str(self.audio_delay),
                "-f",
                "avfoundation",
                "-rtbufsize",
                "256M",
                "-i",
                audio_device,
            ]
            if auto_selected:
                print(
                    "[DirectSbsStream] WARNING: no Stereo Mix device configured; "
                    f"auto-selected macOS audio device {audio_device!r}. If the "
                    "stream has no sound, route system audio to that device or "
                    "pick a Stereo Mix device in the GUI.",
                    flush=True,
                )
            else:
                print(
                    f"[DirectSbsStream] macOS audio device: {device!r} -> "
                    f"avfoundation {audio_device}",
                    flush=True,
                )
            return args
        return []

    def _qsv_d3d11_surface_upload_enabled(self) -> bool:
        """Opt into FFmpeg's Windows D3D11/QSV surface upload boundary.

        Raw RGB stdin is still a host boundary; this mode only keeps the
        post-format frame in a D3D11 hardware frame before h264_qsv/hevc_qsv.
        It is therefore not reported as strict zero-copy.
        """
        return (
            self.os_name == "Windows"
            and self.video_encoder.endswith("_qsv")
            and str(os.environ.get("D2S_QSV_D3D11_UPLOAD", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )

    def _ffmpeg_command(self, width: int, height: int) -> list[str]:
        calibration_stream = getattr(self, "_calibration_controller", None) is not None
        # Calibration must not inherit inference/RGB24 throughput. FFmpeg owns
        # the clock and produces a deterministic 30 FPS pressure stream.
        audio_args = [] if calibration_stream else self._audio_input_args()
        self._active_rate_budget = self._dynamic_stream_rate_budget(width, height)
        target_rate = (
            f"{self._active_rate_budget[0]}M"
            if self._active_rate_budget is not None
            else "0"
        )
        input_args = (
            [
                "-re",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={width}x{height}:rate={self.fps}",
            ]
            if calibration_stream
            else [
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(self.fps),
                # Wall-clock timestamps: the app-paced pipe producer advances
                # video PTS by 1/fps per frame regardless of delivery time,
                # so every stall permanently shifts video PTS behind the
                # real-time audio clock -> growing A/V offset -> the client
                # constantly re-syncs (choppy sound). Timestamping at read
                # time keeps video PTS on the same av_gettime() clock as the
                # audio input (same pattern as the NVIDIA SRT path).
                "-use_wallclock_as_timestamps",
                "1",
                # Threaded demux: the pipe read runs on a worker thread so a
                # stalled/slow app producer can never block the demux loop
                # and starve the real-time audio input (audio dropouts).
                "-thread_queue_size",
                "16",
                "-i",
                "pipe:0",
            ]
        )
        audio_input_args = (
            [
                # Same clock base as the video input (see above), so the
                # muxer sees both streams on one timeline and never has to
                # drop audio that runs ahead of a delayed video PTS.
                "-thread_queue_size",
                "512",
                "-use_wallclock_as_timestamps",
                "1",
                *audio_args,
            ]
            if audio_args
            else []
        )
        command = [
            str(self.ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "64",
            "-analyzeduration",
            "0",
            *input_args,
            *audio_input_args,
            "-map",
            "0:v:0",
        ]
        if audio_args:
            command.extend(["-map", "1:a:0"])
        if self.video_encoder in {"h264_videotoolbox", "hevc_videotoolbox"}:
            command.extend(
                [
                    "-c:v",
                    self.video_encoder,
                    "-profile:v",
                    "main" if self.use_hevc else "high",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",
                    target_rate if self._active_rate_budget is not None else "10M",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-realtime",
                    "true",
                ]
            )
        elif self.video_encoder in {
            "h264_nvenc", "hevc_nvenc",
            "h264_qsv", "hevc_qsv",
            "h264_amf", "hevc_amf",
            "h264_vaapi", "hevc_vaapi",
        }:
            command.extend(
                [
                    "-c:v",
                    self.video_encoder,
                    "-preset",
                    "p1" if self.video_encoder.endswith("_nvenc") else "fast",
                    "-tune",
                    "ll" if self.video_encoder.endswith(("_nvenc", "_qsv")) else "zerolatency",
                    "-rc",
                    "vbr",
                    "-cq",
                    str(self.crf),
                    "-b:v",
                    target_rate,
                    *(
                        ["-vf", "format=nv12,hwupload"]
                        if self.video_encoder.endswith("_vaapi")
                        else []
                    ),
                    "-pix_fmt",
                    "yuv420p",
                    "-bf",
                    "0",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-zerolatency",
                    "1",
                    "-forced-idr",
                    "1",
                    "-strict_gop",
                    "1",
                    "-spatial-aq",
                    "1",
                    "-temporal-aq",
                    "1",
                    "-aq-strength",
                    "8",
                ]
            )
        elif self.video_encoder == "libx265":
            command.extend(
                [
                    "-c:v",
                    "libx265",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                    "-bf",
                    "0",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-crf",
                    str(self.crf),
                    "-x265-params",
                    f"keyint={self.fps}:min-keyint={self.fps}:scenecut=0:"
                    "rc-lookahead=0:open-gop=0:repeat-headers=1",
                ]
            )
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-pix_fmt",
                    "yuv420p",
                    "-bf",
                    "0",
                    "-g",
                    str(self.fps),
                    "-r",
                    str(self.fps),
                    "-crf",
                    str(self.crf),
                    "-x264-params",
                    f"keyint={self.fps}:min-keyint={self.fps}:scenecut=0:"
                    "rc-lookahead=0:open-gop=0:repeat-headers=1",
                ]
            )
        if self.video_encoder.endswith("_vaapi"):
            vaapi_device = os.environ.get("D2S_VAAPI_DEVICE", "/dev/dri/renderD128")
            command[1:1] = ["-vaapi_device", vaapi_device]
            pix_fmt_index = command.index("-pix_fmt")
            command[pix_fmt_index + 1] = "nv12"
            for option in ("-tune", "-rc", "-cq", "-zerolatency", "-forced-idr", "-strict_gop", "-spatial-aq", "-temporal-aq", "-aq-strength"):
                while option in command:
                    index = command.index(option)
                    del command[index:index + 2]
        elif self.video_encoder.endswith("_qsv"):
            for option in ("-tune", "-rc", "-cq", "-zerolatency", "-forced-idr", "-strict_gop", "-spatial-aq", "-temporal-aq", "-aq-strength"):
                while option in command:
                    index = command.index(option)
                    del command[index:index + 2]
            command.extend(["-global_quality", str(self.crf), "-look_ahead", "0"])
            if self._qsv_d3d11_surface_upload_enabled():
                adapter_index = str(os.environ.get("D2S_QSV_D3D11_ADAPTER", "0"))
                command[1:1] = [
                    "-init_hw_device",
                    f"d3d11va=d2s_d3d11:{adapter_index}",
                    "-init_hw_device",
                    "qsv=d2s_qsv@d2s_d3d11",
                    "-filter_hw_device",
                    "d2s_qsv",
                ]
                while "-pix_fmt" in command:
                    pix_fmt_index = command.index("-pix_fmt")
                    del command[pix_fmt_index : pix_fmt_index + 2]
                command.extend(
                    [
                        "-vf",
                        "format=nv12,hwupload=extra_hw_frames=16,"
                        "hwmap=derive_device=qsv,format=qsv",
                    ]
                )
                self._qsv_surface_mode = "d3d11_upload"
            else:
                self._qsv_surface_mode = "host_upload"
        elif self.video_encoder.endswith("_amf"):
            # h264_amf/hevc_amf have no FFmpeg "preset" option (that is
            # NVENC/QSV); the shared hardware branch adds "-preset fast" so it
            # must be stripped here or AMF rejects it at option-apply time
            # ("Error setting option preset to value fast" -> the stream dies
            # right at encoder init). Only the AMF branch is touched; NVENC,
            # QSV, VAAPI, VideoToolbox and libx264/265 keep their presets.
            for option in ("-preset", "-tune", "-rc", "-cq", "-zerolatency", "-forced-idr", "-strict_gop", "-spatial-aq", "-temporal-aq", "-aq-strength"):
                while option in command:
                    index = command.index(option)
                    del command[index:index + 2]
            # AMF's "ultralowlatency" usage only emits ONE real IDR at stream
            # start: -g / -force_key_frames turn into non-IDR I-slices, so a
            # browser WebRTC H.264 depacketizer joining mid-stream never sees
            # a keyframe and shows black forever (framesReceived stays 0 even
            # though all RTP flows - verified via MediaMTX WHEP getStats).
            # "webcam" usage keeps low latency but honors the GOP with true
            # IDR frames, which is what WebRTC browsers require. Only the
            # WebRTC+H.264 AMF path switches; SRT/RTSP headset paths keep
            # ultralowlatency, and NVIDIA/macOS never select "_amf".
            amf_usage = (
                "webcam"
                if self.protocol == "WEBRTC" and not self.use_hevc
                else "ultralowlatency"
            )
            command.extend(["-usage", amf_usage, "-quality", "speed", "-rc", "vbr_peak"])
            if (
                self.protocol == "WEBRTC"
                and not self.use_hevc
                and self.os_name == "Windows"
            ):
                # Browser WebRTC H.264 decoders (Chrome/Edge/Firefox) reject
                # Main-profile streams and enforce the SPS level against the
                # frame rate. AMF auto-selects level 5.1 for 4K, which only
                # supports 4K@~30 fps, so a 4K@40 stream carries an invalid
                # SPS and the browser shows a black frame even though RTP
                # flows (verified: MediaMTX answered the WHEP offer with
                # profile-level-id 42e01f Constrained Baseline while AMF
                # emitted Main 4D0433). Force Constrained Baseline + the
                # level actually required by resolution/fps (e.g. 5.2 for
                # 4K@40) so the SPS matches the negotiated profile and the
                # decoder accepts the stream. H.264 only; HEVC AMF keeps its
                # defaults. NVIDIA/macOS paths never select "_amf".
                required_level = _required_h264_level(width, height, self.fps)
                command.extend(
                    [
                        "-profile:v",
                        "constrained_baseline",
                        "-level:v",
                        _format_h264_level(required_level),
                    ]
                )

        if calibration_stream:
            # Normal playback remains quality-oriented VBR. Calibration uses
            # constant-rate output so each tier applies the requested load to
            # the PC-to-headset network instead of merely changing a VBR cap.
            for option in ("-cq", "-crf", "-global_quality"):
                while option in command:
                    index = command.index(option)
                    del command[index:index + 2]
            if self.video_encoder.endswith("_nvenc"):
                rc_index = command.index("-rc")
                command[rc_index + 1] = "cbr"
            elif self.video_encoder.endswith("_amf"):
                rc_index = command.index("-rc")
                command[rc_index + 1] = "cbr"
            if "-b:v" not in command:
                command.extend(["-b:v", target_rate])
            command.extend(["-minrate", target_rate])

        if self._active_rate_budget is not None:
            target_mbps, peak_mbps, buffer_mbps = self._active_rate_budget
            if calibration_stream:
                peak_mbps = target_mbps
                buffer_mbps = target_mbps
            command.extend(
                [
                    "-maxrate",
                    f"{peak_mbps}M",
                    "-bufsize",
                    f"{buffer_mbps}M",
                ]
            )
        if audio_args:
            # Both inputs share the av_gettime() wall-clock base, so the
            # audio and video timelines are aligned by construction. async=1
            # (the v2.5.0 magnitude) absorbs residual device-clock drift by
            # inserting/dropping samples without letting the filter make
            # large, audible adjustments.
            command.extend(["-af", "aresample=async=1"])
            if self.protocol == "WEBRTC" or self.os_name == "Darwin":
                command.extend(
                    [
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
                # Windows/Linux SRT/RTMP paths (NVIDIA/ROCm) keep AAC; the
                # resample above still normalizes their audio timeline.
                command.extend(["-c:a", "aac", "-ar", "48000", "-b:a", "128k"])
        if getattr(self, "_calibration_controller", None) is not None:
            # FFmpeg reports the actual encoded/muxed output rate, which is
            # different from the configured target bitrate for VBR content.
            command.extend(["-progress", "pipe:2", "-stats_period", "1"])
        if self.os_name == "Windows":
            command.extend(
                [
                    "-force_key_frames",
                    # Frame-based (not t-based): the video input now carries
                    # wall-clock PTS, so a time expression like
                    # expr:gte(t,n_forced*1) would force every frame to be a
                    # keyframe. n is the frame index, independent of the PTS
                    # base; one keyframe per fps frames (1/s cadence). Use
                    # mod() (not the % operator, which the bundled FFmpeg's
                    # force_key_frames evaluator rejects as "Missing ')' or
                    # too many args") and no backslash-escaping (the command
                    # is spawned as an argv list, so FFmpeg receives the
                    # expression verbatim).
                    f"expr:eq(mod(n,{self.fps}),0)",
                    "-muxdelay",
                    "0",
                    "-muxpreload",
                    "0",
                    "-flush_packets",
                    "1",
                    # Zero waits indefinitely for every stream and can starve RTSP.
                    "-max_interleave_delta",
                    "100000",
                ]
            )
            if self.protocol == "WEBRTC":
                # Publish WebRTC's source through the local RTSP listener.
                # This avoids waiting for SRT MPEG-TS stream discovery while
                # MediaMTX still performs the RTSP→WebRTC conversion.
                command.extend(
                    [
                        "-f",
                        "rtsp",
                        "-rtsp_transport",
                        "tcp",
                        "-pkt_size",
                        "1452",
                        f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}"
                        "?pkt_size=1452",
                    ]
                )
            else:
                command.extend(
                    [
                        "-f",
                        "mpegts",
                        "-mpegts_flags",
                        "+resend_headers",
                        f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}&pkt_size=1316",
                    ]
                )
        else:
            command.extend(
                [
                    "-threads",
                    "2",
                    # Bound sparse-stream interleaving to 100 ms.
                    "-max_interleave_delta",
                    "100000",
                    "-f",
                    "rtsp",
                    "-rtsp_transport",
                    "tcp",
                    "-pkt_size",
                    "1452",
                    f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}"
                    "?pkt_size=1452",
                ]
            )
        return command

    def _probe_darwin_audio_silence(self) -> None:
        """Warn once when the macOS audio device captures digital silence.

        Runs a short volumedetect capture on the same AVFoundation device the
        stream uses (CoreAudio allows concurrent capture clients). Purely
        advisory: a silent result never fails or delays the stream. macOS
        only; other platforms are no-ops.
        """
        if getattr(self, "os_name", None) != "Darwin":
            return
        device = getattr(self, "_darwin_audio_device", None)
        if not device:
            return
        ffmpeg_path = getattr(self, "ffmpeg_path", None)
        if not ffmpeg_path:
            return
        try:
            result = subprocess.run(
                [
                    str(ffmpeg_path),
                    "-hide_banner",
                    "-f",
                    "avfoundation",
                    "-rtbufsize",
                    "256M",
                    "-i",
                    device,
                    "-t",
                    "1",
                    "-af",
                    "volumedetect",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=2.5,
            )
        except (OSError, subprocess.SubprocessError):
            return
        match = re.search(r"mean_volume:\s*(-?[0-9.]+)\s*dB", result.stderr or "")
        if match is None:
            return
        mean_db = float(match.group(1))
        if mean_db < -55.0:
            print(
                "[DirectSbsStream] WARNING: macOS audio device "
                f"{device!r} appears silent (mean_volume={mean_db:.1f} dB). "
                "Route system audio to that device, or select a different "
                "Stereo Mix device in the GUI so the stream carries sound.",
                flush=True,
            )

    def _start_ffmpeg(self, width: int, height: int) -> None:
        # Audio: old main.py used dshow directly (ffmpeg -f dshow -i audio={device}) with no broken sound.
        # Previous wasapi UDP loopback (SoundcardLoopbackSender) caused fragmentation/discontinuity -> broken audio.
        # Kept dshow path in _audio_input_args, so disable python loopback start here.
        if False and (
            self._calibration_controller is None
            and self.os_name == "Windows"
            and self.stereo_mix_device.casefold().startswith("soundcard:")
        ):
            try:
                device_name = self.stereo_mix_device.split(":", 1)[1].strip()
                self._soundcard_audio = SoundcardLoopbackSender(device_name or None)
                self._soundcard_audio.start()
            except Exception as exc:
                print(
                    f"[DirectSbsStream] soundcard loopback unavailable: {exc}; "
                    "falling back to dshow",
                    flush=True,
                )
                self._soundcard_audio = None
                self.stereo_mix_device = device_name
        if not self._encoder_selected:
            self.video_encoder = self._select_video_encoder(width, height)
            self._encoder_selected = True
        command = self._ffmpeg_command(width, height)
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if self.os_name == "Windows"
            else 0
        )
        self.ffmpeg_process = subprocess.Popen(
            command,
            stdin=(subprocess.DEVNULL if self._calibration_controller is not None else subprocess.PIPE),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._ffmpeg_stderr_tail = []
        self._ffmpeg_log_thread = threading.Thread(
            target=self._drain_ffmpeg_stderr,
            args=(self.ffmpeg_process,),
            name="DirectSbsFfmpegLog",
            daemon=True,
        )
        self._ffmpeg_log_thread.start()
        time.sleep(0.05)
        if self.ffmpeg_process.poll() is not None:
            detail = "; ".join(self._ffmpeg_stderr_tail[-3:]) or "no FFmpeg diagnostic"
            if (
                self.os_name == "Windows"
                and self.stereo_mix_device
                and not self._audio_startup_retried
                and re.search(
                    r"\[in#[0-9]+\].*error opening input|"
                    r"(dshow|wasapi).*(i/o error|no such device|device not found)|"
                    r"i/o error.*(dshow|wasapi|audio)",
                    detail,
                    re.IGNORECASE,
                )
            ):
                # The configured audio capture device cannot be opened (missing
                # "Stereo Mix" loopback, unplugged device, wrong name). FFmpeg
                # dies at startup, so retry once with audio disabled instead of
                # failing the whole stream; the video-only stream still starts.
                print(
                    f"[DirectSbsStream] audio input failed to open ({detail}); "
                    "retrying without audio",
                    flush=True,
                )
                self._audio_startup_retried = True
                self.stereo_mix_device = ""
                self._stop_process(self.ffmpeg_process)
                self.ffmpeg_process = None
                self._frame_size = None
                self._start_ffmpeg(width, height)
                return
            raise RuntimeError(
                f"FFmpeg exited during startup with code {self.ffmpeg_process.returncode}: {detail}"
            )
        self._frame_size = (width, height)
        if (
            self.os_name == "Darwin"
            and not getattr(self, "_darwin_audio_probe_started", False)
            and getattr(self, "_darwin_audio_device", None)
        ):
            # Advisory silence check on the captured device, once per stream
            # lifecycle; runs off the pipeline thread so it never blocks
            # frame submission.
            self._darwin_audio_probe_started = True

            def _delayed_probe() -> None:
                time.sleep(0.5)
                self._probe_darwin_audio_silence()

            threading.Thread(
                target=_delayed_probe,
                name="DarwinAudioSilenceProbe",
                daemon=True,
            ).start()
        if self._active_rate_budget is not None:
            target_mbps, peak_mbps, buffer_mbps = self._active_rate_budget
            print(
                f"[DirectSbsStream] Dynamic stream quality: protocol={self.protocol} "
                f"target={target_mbps}M "
                f"peak={peak_mbps}M buffer={buffer_mbps}M "
                f"resolution={width}x{height} fps={self.fps} crf={self.crf}",
                flush=True,
            )
        if self._calibration_controller is not None:
            print(
                f"[StreamCalibration] Independent CBR pressure stream active: "
                f"{width}x{height}@{self.fps} encoder={self.video_encoder}",
                flush=True,
            )
        else:
            print(
                f"[DirectSbsStream] FFmpeg consumes RGB24 SBS directly: "
                f"{width}x{height}@{self.fps} encoder={self.video_encoder} "
                f"qsv_surface={self._qsv_surface_mode} "
                f"gpu_to_cpu=True zero_copy=False "
                f"gpu_copy_count={1 if self._qsv_surface_mode == 'd3d11_upload' else 0}",
                flush=True,
            )

    def _write_frame(self, frame: np.ndarray) -> None:
        process = self.ffmpeg_process
        if process is None or process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        if process.poll() is not None:
            detail = "; ".join(self._ffmpeg_stderr_tail[-3:]) or "no FFmpeg diagnostic"
            raise RuntimeError(
                f"FFmpeg exited with code {process.returncode}: {detail}"
            )
        process.stdin.write(memoryview(frame).cast("B"))
        process.stdin.flush()

    def _drain_ffmpeg_stderr(self, process: subprocess.Popen) -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            for raw_line in stream:
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                else:
                    line = str(raw_line).strip()
                if not line:
                    continue
                match = re.match(
                    r"bitrate=\s*([0-9.]+)([kmg]?bits/s)",
                    line,
                    re.IGNORECASE,
                )
                if match:
                    value = float(match.group(1))
                    unit = match.group(2).casefold()
                    multiplier = {
                        "bits/s": 1e-6,
                        "kbits/s": 1e-3,
                        "mbits/s": 1.0,
                    }[unit]
                    self._ffmpeg_bitrate_mbps = value * multiplier
                self._ffmpeg_stderr_tail.append(line)
                del self._ffmpeg_stderr_tail[:-20]
                if any(token in line.casefold() for token in ("error", "failed", "invalid", "cannot")):
                    print(f"[DirectSbsStream] FFmpeg: {line}", flush=True)
        except (OSError, ValueError):
            return

    def request_audio_delay(self, delay: float) -> bool:
        delay = max(-10.0, min(10.0, float(delay)))
        with self._audio_delay_lock:
            current = (
                self._pending_audio_delay
                if self._pending_audio_delay is not None
                else self.audio_delay
            )
            if math.isclose(delay, current, abs_tol=1e-6):
                return False
            self._pending_audio_delay = delay
        return True

    def _apply_pending_audio_delay(self) -> None:
        with self._audio_delay_lock:
            delay = self._pending_audio_delay
            self._pending_audio_delay = None
        if delay is None or math.isclose(delay, self.audio_delay, abs_tol=1e-6):
            return
        previous = self.audio_delay
        self.audio_delay = delay
        if self.ffmpeg_process is None:
            return
        print(
            f"[DirectSbsStream] Audio delay changed: {previous:.3f}s -> "
            f"{delay:.3f}s; restarting FFmpeg publisher",
            flush=True,
        )
        self._stop_process(self.ffmpeg_process)
        self.ffmpeg_process = None
        self._frame_size = None

    def submit_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        size = (int(width), int(height))
        self._apply_pending_audio_delay()
        if self.ffmpeg_process is None:
            self._start_ffmpeg(*size)
        elif self._frame_size != size:
            raise RuntimeError(
                f"SBS stream size changed from {self._frame_size} to {size}; restart required"
            )
        if self._calibration_controller is not None:
            # The independent lavfi source runs continuously at 30 FPS. Runtime
            # frames are intentionally ignored so RGB conversion/inference
            # speed cannot throttle the bandwidth probe.
            return
        try:
            self._write_frame(frame)
        except (BrokenPipeError, OSError, RuntimeError) as exc:
            # A dead FFmpeg whose last diagnostics point at the audio input
            # (dshow/wasapi open failure) is restarted once without audio so
            # the stream still starts video-only. This guards the case where
            # the audio open fails after the short startup probe window.
            if (
                self.os_name == "Windows"
                and self.stereo_mix_device
                and not self._audio_startup_retried
                and re.search(
                    r"\[in#[0-9]+\].*error opening input|"
                    r"(dshow|wasapi).*(i/o error|no such device|device not found)|"
                    r"i/o error.*(dshow|wasapi|audio)",
                    str(exc),
                    re.IGNORECASE,
                )
            ):
                print(
                    f"[DirectSbsStream] audio input failed during startup "
                    f"({exc}); retrying without audio",
                    flush=True,
                )
                self._audio_startup_retried = True
                self.stereo_mix_device = ""
                self._stop_process(self.ffmpeg_process)
                self.ffmpeg_process = None
                self._frame_size = None
                self._start_ffmpeg(*size)
                self._write_frame(frame)
                return
            if self.video_encoder not in {"h264_nvenc", "hevc_nvenc"}:
                raise
            software_encoder = "libx265" if self.use_hevc else "libx264"
            print(
                f"[DirectSbsStream] NVENC startup failed; retrying with "
                f"{software_encoder}",
                flush=True,
            )
            self._stop_process(self.ffmpeg_process)
            self.ffmpeg_process = None
            self._frame_size = None
            self.video_encoder = software_encoder
            self._start_ffmpeg(*size)
            self._write_frame(frame)

    @staticmethod
    def _stop_process(process: subprocess.Popen | None) -> None:
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except Exception:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)

    def close(self) -> None:
        self._mediamtx_metrics_stop.set()
        if self._mediamtx_metrics_thread is not None:
            self._mediamtx_metrics_thread.join(timeout=1.5)
            self._mediamtx_metrics_thread = None
        if self._calibration_controller is not None:
            self._calibration_controller.close()
            self._calibration_controller = None
        if self._soundcard_audio is not None:
            self._soundcard_audio.close()
            self._soundcard_audio = None
        self._stop_process(self.ffmpeg_process)
        self._stop_process(self.server_process)
        if self._server_log_thread is not None:
            self._server_log_thread.join(timeout=0.5)
        self.ffmpeg_process = None
        self.server_process = None
        self._server_log_thread = None
        self._ffmpeg_log_thread = None
        self._ffmpeg_stderr_tail = []
        # Ensure orphan MediaMTX/FFmpeg ports are freed on stop (previous run left :8000/:9998 occupied)
        try:
            _kill_orphan_mediamtx_processes(ports=[self.port, 9998, 8000, 8001, 8189, 8554, 1935, 8888, 8889, 8890])
        except Exception:
            pass


class IntelQsvDirectSbsOutput(FfmpegDirectSbsOutput):
    """Intel final-SBS path using FFmpeg's D3D11-to-QSV surface boundary.

    The final SBS frame is currently produced as CPU RGB by the stereo
    runtime. This backend deliberately makes that boundary visible: FFmpeg
    uploads the RGB frame to a D3D11 surface and derives a QSV surface for
    hardware encoding. It is a GPU-accelerated Intel path, but not strict
    capture-to-encode zero-copy until the stereo compositor exports a native
    D3D11/Vulkan surface.
    """

    def _encoder_candidates(self) -> list[tuple[str, str]]:
        codec = "hevc" if self.use_hevc else "h264"
        return [
            (f"{codec}_qsv", "Intel Quick Sync/D3D11"),
            ("libx265" if self.use_hevc else "libx264", "software fallback"),
        ]

    def _qsv_d3d11_surface_upload_enabled(self) -> bool:
        configured = os.environ.get("D2S_QSV_D3D11_UPLOAD", "1")
        return (
            self.os_name == "Windows"
            and self.video_encoder.endswith("_qsv")
            and str(configured).strip().casefold()
            in {"1", "true", "yes", "on"}
        )

    def _start_ffmpeg(self, width: int, height: int) -> None:
        super()._start_ffmpeg(width, height)
        if self.video_encoder.endswith("_qsv"):
            print(
                "[IntelStream] final SBS path: RGB24 -> D3D11 NV12 -> QSV "
                "surface; gpu_to_cpu=True zero_copy=False gpu_copy_count=1",
                flush=True,
            )


class VulkanDirectSbsOutput(FfmpegDirectSbsOutput):
    """GPU image path: CUDA RGBA import, native Vulkan conversion/encode, RTSP/SRT mux."""

    synchronous_submit = False

    def __init__(self, *args, **kwargs) -> None:
        self._vendor_gpu_first = bool(kwargs.pop("vendor_gpu_first", False))
        super().__init__(*args, **kwargs)
        self._native_vulkan_bridge = None
        self._native_vulkan_dll_dir = None
        self._native_vulkan_encoder = None
        self._native_vulkan_importer = None
        self._native_mux_process: subprocess.Popen | None = None
        self._native_active = False
        self._native_pts = 0
        self._opengl_fallback: OpenGLFallbackBackend | None = None
        self._opengl_pynv_fallback: PyNvDirectSbsOutput | None = None
        self._opengl_nvenc_mode = "none"
        self._opengl_amd_fallback: AmdAmfDirectSbsOutput | None = None
        self._opengl_fallback_active = False
        self._opengl_fallback_attempted = False
        self._native_vulkan_load_attempted = False
        if not self._vendor_gpu_first:
            self._load_native_vulkan_bridge()

    def _load_native_vulkan_bridge(self) -> None:
        if self._native_vulkan_load_attempted:
            return
        self._native_vulkan_load_attempted = True
        try:
            validation_layers = os.environ.get("VK_INSTANCE_LAYERS", "")
            if "VK_LAYER_KHRONOS_validation" in validation_layers:
                print(
                    "[VulkanStream] native bridge disabled under "
                    "VK_LAYER_KHRONOS_validation",
                    flush=True,
                )
                return
            if self.os_name == "Windows" and hasattr(os, "add_dll_directory"):
                self._native_vulkan_dll_dir = os.add_dll_directory(
                    str(Path(self.ffmpeg_path).parent)
                )
            self._native_vulkan_bridge = VulkanNativeBridge.load()
            print(
                "[VulkanStream] native bridge loaded; GPU->CPU download=False",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[VulkanStream] native bridge unavailable: {exc}",
                flush=True,
            )

    def _select_video_encoder(self, width: int, height: int) -> str:
        report = probe_vulkan_video(
            self.ffmpeg_path,
            width=width,
            height=height,
            hevc=self.use_hevc,
            timeout=float(os.environ.get("D2S_VULKAN_PROBE_TIMEOUT", "8")),
            os_name=self.os_name,
        )
        if not report.available:
            # The native Vulkan encoder (h264_vulkan/hevc_vulkan) is not usable
            # on this device (e.g. AMD LLPC). Fall back through the normal
            # vendor -> software chain instead of crashing the stream with the
            # raw FFmpeg probe error. Working NVIDIA/macOS paths are untouched:
            # they succeed this probe and never reach the fallback.
            print(
                f"[VulkanStream] Vulkan encoder unavailable ({report.detail}); "
                "falling back to the vendor/FFmpeg encoder chain",
                flush=True,
            )
            return super()._select_video_encoder(width, height)
        print(
            f"[VulkanStream] Vulkan capability probe: encoder={report.encoder} "
            f"input={report.input_format} {width}x{height}",
            flush=True,
        )
        return report.encoder

    def _ffmpeg_command(self, width: int, height: int) -> list[str]:
        """Keep the validated host-upload command for explicit fallback/diagnostics."""
        command = super()._ffmpeg_command(width, height)
        encoder = getattr(self, "video_encoder", "h264_vulkan")
        command[1:1] = [
            "-init_hw_device",
            "vulkan=d2s_vk:0",
            "-filter_hw_device",
            "d2s_vk",
        ]
        if "-c:v" in command:
            command[command.index("-c:v") + 1] = encoder
        for option in ("-crf", "-x264-params", "-x265-params"):
            while option in command:
                index = command.index(option)
                del command[index:index + 2]
        if "-pix_fmt" in command:
            command[command.index("-pix_fmt") + 1] = "vulkan"
        output_url = command.pop()
        command.extend(
            [
                "-vf",
                "format=nv12,hwupload",
                "-profile:v",
                "main" if getattr(self, "use_hevc", False) else "high",
                *(["-level:v", "5.1"] if not getattr(self, "use_hevc", False) and int(width) >= 2560 else []),
                "-rc_mode",
                "vbr",
                "-qp",
                str(getattr(self, "crf", 23)),
                "-tune",
                "ull",
                "-usage",
                "stream",
                "-bf",
                "0",
            ]
        )
        command.append(output_url)
        return command

    def _native_output_url(self) -> str:
        if self.protocol == "WEBRTC":
            return (
                f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}"
                "?pkt_size=1452"
            )
        return (
            f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}"
            "&pkt_size=1316"
        )

    def _start_native_mux(self) -> None:
        audio_args = self._audio_input_args()
        audio_input_args = (
            [
                # Same clock base and demux decoupling as the FFmpeg path:
                # wall-clock timestamps keep the muxed video PTS on the
                # real-time audio clock, and the audio demux thread can never
                # be starved by a stalled video pipe producer.
                "-thread_queue_size",
                "512",
                "-use_wallclock_as_timestamps",
                "1",
                *audio_args,
            ]
            if audio_args
            else []
        )
        command = [
            str(self.ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-fpsprobesize",
            "0",
            "-f",
            "h264",
            "-r",
            str(self.fps),
            "-use_wallclock_as_timestamps",
            "1",
            "-thread_queue_size",
            "16",
            "-i",
            "pipe:0",
            *audio_input_args,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
        ]
        if audio_args:
            command.extend(["-map", "1:a:0"])
            # Same normalized audio timeline as the FFmpeg path (async=1,
            # the v2.5.0 magnitude) for every client protocol.
            command.extend(["-af", "aresample=async=1"])
            if self.protocol == "WEBRTC":
                command.extend(
                    [
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
            ]
        )
        if self.protocol == "WEBRTC":
            command.extend(
                [
                    "-f",
                    "rtsp",
                    "-rtsp_transport",
                    "tcp",
                    "-pkt_size",
                    "1452",
                ]
            )
        else:
            command.extend(["-f", "mpegts", "-mpegts_flags", "+resend_headers"])
        command.append(self._native_output_url())
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if self.os_name == "Windows"
            else 0
        )
        self._native_mux_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._ffmpeg_stderr_tail = []
        self._ffmpeg_log_thread = threading.Thread(
            target=self._drain_ffmpeg_stderr,
            args=(self._native_mux_process,),
            name="DirectSbsVulkanMuxLog",
            daemon=True,
        )
        self._ffmpeg_log_thread.start()
        time.sleep(0.05)
        if self._native_mux_process.poll() is not None:
            detail = "; ".join(self._ffmpeg_stderr_tail[-3:]) or "no FFmpeg diagnostic"
            raise RuntimeError(
                f"Vulkan packet muxer exited with code "
                f"{self._native_mux_process.returncode}: {detail}"
            )

    @staticmethod
    def _normalize_device_uuid(value: Any) -> str:
        if isinstance(value, bytes):
            return value.hex().casefold()
        return "".join(
            character for character in str(value or "").casefold()
            if character.isalnum()
        )

    def _verify_native_device(self, cuda_device: Any) -> None:
        identity = self._native_vulkan_encoder.device_identity()
        if identity is None:
            print(
                "[VulkanStream] Vulkan/CUDA device UUID query unavailable; "
                "native path continues with single-device compatibility mode",
                flush=True,
            )
            return
        native_uuid, native_name = identity
        import torch

        cuda_properties = torch.cuda.get_device_properties(cuda_device)
        cuda_uuid = getattr(cuda_properties, "uuid", None)
        native_key = self._normalize_device_uuid(native_uuid)
        cuda_key = self._normalize_device_uuid(cuda_uuid)
        if not native_key or not cuda_key or native_key != cuda_key:
            raise RuntimeError(
                "CUDA/Vulkan physical-device UUID mismatch: "
                f"CUDA={cuda_uuid!s} Vulkan={native_name or 'unknown'}:{native_uuid.hex()}"
            )
        print(
            f"[VulkanStream] device matched: CUDA {cuda_device} <-> Vulkan "
            f"{native_name or 'unknown'} uuid={native_uuid.hex()}",
            flush=True,
        )

    def _start_native(self, width: int, height: int, *, cuda_device: Any = None) -> None:
        self._load_native_vulkan_bridge()
        if self._native_vulkan_bridge is None:
            raise RuntimeError("native Vulkan FFmpeg bridge is unavailable")
        # See _start_ffmpeg above: disable python wasapi loopback for same broken-audio reason; use dshow
        if False and (
            self._calibration_controller is None
            and self.os_name == "Windows"
            and self.stereo_mix_device.casefold().startswith("soundcard:")
        ):
            device_name = self.stereo_mix_device.split(":", 1)[1].strip()
            self._soundcard_audio = SoundcardLoopbackSender(device_name or None)
            self._soundcard_audio.start()
        self._active_rate_budget = self._dynamic_stream_rate_budget(width, height)
        target_mbps, peak_mbps = (
            self._active_rate_budget[:2]
            if self._active_rate_budget is not None
            else (10, 12)
        )
        self._native_vulkan_encoder = self._native_vulkan_bridge.create_encoder(
            width=width,
            height=height,
            fps=self.fps,
            target_bitrate=int(target_mbps) * 1_000_000,
            peak_bitrate=int(peak_mbps) * 1_000_000,
            hevc=self.use_hevc,
        )
        if cuda_device is not None:
            self._verify_native_device(cuda_device)
        from viewer.cuda_vulkan_interop import CudaVulkanImageImporter

        self._native_vulkan_importer = CudaVulkanImageImporter()
        self._start_native_mux()
        self._native_active = True
        self._native_pts = 0
        self._frame_size = (int(width), int(height))
        print(
            f"[VulkanStream] native GPU image path active: "
            f"CUDA RGBA -> Vulkan Compute RGBA->NV12 -> "
            f"{'hevc_vulkan' if self.use_hevc else 'h264_vulkan'} -> MediaMTX; "
            f"input=RGBA8 encode=NV12 gpu_to_cpu=False "
            f"gpu_copy=True zero_copy=False resolution={width}x{height} "
            f"fps={self.fps} target={target_mbps}M peak={peak_mbps}M",
            flush=True,
        )
        print("[D2S_STATUS] Vulkan native GPU image path active", flush=True)

    @staticmethod
    def _rgba_tensor(frame: Any):
        import torch

        image = getattr(frame, "sbs", frame)
        if not isinstance(image, torch.Tensor) or not bool(image.is_cuda):
            raise RuntimeError("native Vulkan path requires a CUDA tensor")
        if image.ndim == 4:
            if int(image.shape[0]) != 1:
                raise RuntimeError("native Vulkan path accepts one frame")
            image = image[0]
        if image.ndim != 3:
            raise RuntimeError(f"unsupported CUDA SBS shape: {tuple(image.shape)!r}")
        if int(image.shape[0]) in (3, 4):
            image = image.permute(1, 2, 0)
        if int(image.shape[-1]) not in (3, 4):
            raise RuntimeError(f"native Vulkan path requires RGB/RGBA: {tuple(image.shape)!r}")
        if image.dtype != torch.uint8:
            image = image.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        if int(image.shape[-1]) == 3:
            image = torch.cat(
                (image, torch.full((*image.shape[:2], 1), 255, dtype=torch.uint8, device=image.device)),
                dim=-1,
            )
        return image.contiguous()

    def _stop_native(self) -> None:
        self._native_active = False
        if self._native_vulkan_encoder is not None:
            # Every encoded packet is drained immediately after submit. Do not
            # call avcodec_send_frame(NULL) during normal shutdown: FFmpeg's
            # Vulkan Video flush can block under the Khronos validation layer
            # while its internal multi-plane frame pool is being torn down.
            # The muxer is stopped next, so a discarded final drain packet
            # cannot reach the client anyway; close() still releases all native
            # Vulkan resources through the bridge.
            try:
                self._native_vulkan_encoder.close()
            except Exception:
                pass
            self._native_vulkan_encoder = None
        if self._native_vulkan_importer is not None:
            try:
                self._native_vulkan_importer.close()
            except Exception:
                pass
            self._native_vulkan_importer = None
        self._stop_process(self._native_mux_process)
        self._native_mux_process = None

    def _new_host_fallback(self) -> FfmpegDirectSbsOutput:
        fallback = FfmpegDirectSbsOutput(
            base_dir=self.base_dir,
            protocol=self.protocol,
            port=self.port,
            stream_key=self.stream_key,
            fps=self.fps,
            crf=self.crf,
            stereo_mix_device=self.stereo_mix_device,
            audio_delay=self.audio_delay,
            os_name=self.os_name,
            prefer_nvenc=self.prefer_nvenc,
            display_mode=self.display_mode,
            target_bitrate_mbps=self.target_bitrate_mbps,
            peak_bitrate_mbps=self.peak_bitrate_mbps,
            auto_calibration=False,
            calibration_port=self.calibration_port,
            on_calibration_fps=self._on_calibration_fps,
            calibration_fingerprint=None,
        )
        fallback.server_process = self.server_process
        return fallback

    def _new_opengl_pynv_fallback(self) -> PyNvDirectSbsOutput:
        fallback = PyNvDirectSbsOutput(
            base_dir=self.base_dir,
            protocol=self.protocol,
            port=self.port,
            stream_key=self.stream_key,
            fps=self.fps,
            crf=self.crf,
            stereo_mix_device=self.stereo_mix_device,
            audio_delay=self.audio_delay,
            os_name=self.os_name,
            prefer_nvenc=self.prefer_nvenc,
            display_mode=self.display_mode,
            target_bitrate_mbps=self.target_bitrate_mbps,
            peak_bitrate_mbps=self.peak_bitrate_mbps,
            auto_calibration=False,
            calibration_port=self.calibration_port,
            on_calibration_fps=self._on_calibration_fps,
            calibration_fingerprint=None,
        )
        fallback.server_process = self.server_process
        return fallback

    def _new_opengl_amd_fallback(self) -> AmdAmfDirectSbsOutput:
        fallback = AmdAmfDirectSbsOutput(
            base_dir=self.base_dir,
            protocol=self.protocol,
            port=self.port,
            stream_key=self.stream_key,
            fps=self.fps,
            crf=self.crf,
            stereo_mix_device=self.stereo_mix_device,
            audio_delay=self.audio_delay,
            os_name=self.os_name,
            prefer_nvenc=self.prefer_nvenc,
            display_mode=self.display_mode,
            target_bitrate_mbps=self.target_bitrate_mbps,
            peak_bitrate_mbps=self.peak_bitrate_mbps,
            auto_calibration=False,
            calibration_port=self.calibration_port,
            on_calibration_fps=self._on_calibration_fps,
            calibration_fingerprint=None,
        )
        fallback.server_process = self.server_process
        return fallback

    @staticmethod
    def _close_secondary_output(output: Any) -> None:
        if output is None:
            return
        # The Vulkan owner keeps MediaMTX alive. A secondary output must release
        # only its encoder/audio resources during a path transition.
        try:
            output.server_process = None
        except Exception:
            pass
        try:
            output.close()
        except Exception:
            pass

    def _fallback_to_opengl(
        self,
        frame: Any,
        reason: Exception,
        *,
        vendor_first: bool = False,
    ) -> bool:
        if self._opengl_fallback_attempted:
            if not vendor_first:
                self._fallback_to_host(frame, reason)
            return False
        self._opengl_fallback_attempted = True
        enabled = os.environ.get("D2S_OPENGL_FALLBACK", "1").strip().casefold()
        if enabled in {"0", "false", "off", "no"}:
            if not vendor_first:
                self._fallback_to_host(frame, reason)
            return False
        try:
            image = getattr(frame, "sbs", frame)
            is_cuda = bool(getattr(image, "is_cuda", False))
            if is_cuda:
                import torch

                if image.ndim == 4:
                    image = image[0]
                if image.ndim != 3:
                    raise RuntimeError(f"unsupported CUDA SBS shape: {tuple(image.shape)!r}")
                height, width = (
                    (int(image.shape[-2]), int(image.shape[-1]))
                    if int(image.shape[0]) in (1, 3, 4)
                    else (int(image.shape[0]), int(image.shape[1]))
                )
                del torch
            else:
                rgb = runtime_sbs_to_rgb(frame)
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
            backend = OpenGLFallbackBackend(width, height, pbo_count=3)
            caps = backend.capabilities
            print(
                f"[OpenGLStream] fallback candidate: context={caps.context_api} "
                f"texture={caps.texture_format} framebuffer="
                f"{int(getattr(caps, 'framebuffer_supported', False))} "
                f"pbo={caps.pbo_count} fence={int(caps.fence_supported)} "
                f"cuda_gl_interop={caps.cuda_gl_interop} "
                f"hip_gl_interop={getattr(caps, 'hip_gl_interop', False)} "
                f"gpu_copy_count={getattr(caps, 'gpu_copy_count', 0)} "
                f"interop={getattr(caps, 'interop_mode', 'none')}",
                flush=True,
            )
            self._opengl_fallback = backend
            self._opengl_fallback_active = True
            if is_cuda and caps.cuda_gl_interop:
                fallback = self._new_opengl_pynv_fallback()
                self._opengl_pynv_fallback = fallback
                try:
                    surface_view = backend.submit_cuda_tensor_surface(frame)
                    fallback._start_cudaarray_encoder(
                        width, height, surface_view.cuda_array
                    )
                    assert fallback._pynv_output is not None
                    fallback._pynv_output.submit_cuda_frame(surface_view)
                    self._opengl_nvenc_mode = "direct"
                    print(
                        "[OpenGLStream] active: "
                        "encoder=NativeNVENC/CUDAARRAY-SurfaceKernel "
                        "gpu_to_cpu=False zero_copy=True gpu_copy_count=0 "
                        f"resolution={width}x{height} fps={self.fps}",
                        flush=True,
                    )
                except Exception as direct_exc:
                    fallback._release_pynv_pipeline()
                    backend.release_cuda_array()
                    print(
                        "[OpenGLStream] direct CUDA surface write unavailable: "
                        f"{type(direct_exc).__name__}: {direct_exc}; "
                        "fallback=CUDAARRAY device copy",
                        flush=True,
                    )
                    try:
                        cuda_array = backend.submit_cuda_array(frame)
                        fallback._start_cudaarray_encoder(
                            width, height, cuda_array
                        )
                        assert fallback._pynv_output is not None
                        fallback._pynv_output.submit_cuda_frame(cuda_array)
                        self._opengl_nvenc_mode = "array-copy"
                        print(
                            "[OpenGLStream] active: "
                            "encoder=NativeNVENC/CUDAARRAY "
                            "gpu_to_cpu=False zero_copy=False gpu_copy_count=1 "
                            f"resolution={width}x{height} fps={self.fps}",
                            flush=True,
                        )
                    except Exception as native_exc:
                        fallback._release_pynv_pipeline()
                        backend.release_cuda_array()
                        print(
                            "[OpenGLStream] native NVENC CUDAARRAY unavailable: "
                            f"{type(native_exc).__name__}: {native_exc}; "
                            "fallback=PyNvVideoCodec",
                            flush=True,
                        )
                        fallback._start_ffmpeg(width, height)
                        assert fallback._pynv_output is not None
                        fallback._pynv_output.submit_cuda_frame(
                            backend.submit_cuda(frame)
                        )
                        self._opengl_nvenc_mode = "pynv"
                        print(
                            "[OpenGLStream] active: encoder=PyNvVideoCodec/NVENC "
                            f"gpu_to_cpu={caps.gpu_to_cpu} "
                            f"zero_copy={caps.zero_copy} "
                            f"gpu_copy_count={getattr(caps, 'gpu_copy_count', 0)} "
                            f"resolution={width}x{height} fps={self.fps}",
                            flush=True,
                        )
            elif is_cuda and caps.hip_gl_interop:
                if self.stereo_mix_device:
                    raise RuntimeError(
                        "OpenGL HIP/AMF fallback requires audio-disabled mode; "
                        "use stable FFmpeg audio path"
                    )
                fallback = self._new_opengl_amd_fallback()
                self._opengl_amd_fallback = fallback
                rgba = backend.submit_cuda(frame)
                fallback._start_amd_encoder(rgba)
                fallback._submit_amd_packet(rgba)
                print(
                    "[OpenGLStream] active: encoder=AMF "
                    f"gpu_to_cpu={caps.gpu_to_cpu} zero_copy={caps.zero_copy} "
                    f"gpu_copy_count={getattr(caps, 'gpu_copy_count', 0)} "
                    f"resolution={width}x{height} fps={self.fps}",
                    flush=True,
                )
            elif vendor_first:
                raise RuntimeError(
                    "no native CUDA/OpenGL/NVENC or HIP/OpenGL/AMF interop "
                    "is available for the vendor-first stage"
                )
            else:
                # OpenGL has no portable resource-sharing ABI with QSV,
                # VAAPI, or VideoToolbox. Once the context/PBO/fence probe
                # succeeds, keep the stable host-upload path and do not upload
                # the already-CPU RGB frame into GL merely to read it back.
                if is_cuda:
                    rgb = runtime_sbs_to_rgb(frame)
                fallback = self._new_host_fallback()
                self._host_fallback = fallback
                fallback.submit_frame(rgb)
                selected_encoder = getattr(fallback, "video_encoder", "unknown")
                print(
                    f"[OpenGLStream] active: encoder=FFmpeg/{selected_encoder} "
                    "interop=none gpu_to_cpu=True zero_copy=False "
                    "gpu_copy_count=0 "
                    f"resolution={width}x{height} fps={self.fps}",
                    flush=True,
                )
            if vendor_first:
                print("[D2S_STATUS] Native vendor GPU streaming active", flush=True)
            else:
                print("[D2S_STATUS] Vulkan unavailable; using OpenGL fallback", flush=True)
            return True
        except Exception as exc:
            next_stage = "Vulkan" if vendor_first else "stable advanced FFmpeg path"
            print(
                f"[OpenGLStream] unavailable: {type(exc).__name__}: {exc}; "
                f"fallback={next_stage}",
                flush=True,
            )
            self._close_secondary_output(self._opengl_pynv_fallback)
            self._opengl_pynv_fallback = None
            self._close_secondary_output(self._opengl_amd_fallback)
            self._opengl_amd_fallback = None
            if self._opengl_fallback is not None:
                self._opengl_fallback.close()
            self._opengl_fallback = None
            self._opengl_fallback_active = False
            if vendor_first:
                print(
                    "[D2S_STATUS] Native vendor GPU unavailable; trying Vulkan",
                    flush=True,
                )
                return False
            self._fallback_to_host(frame, exc)
            return False

    def _fallback_to_host(self, frame: Any, reason: Exception) -> None:
        print(
            f"[VulkanStream] native GPU path failed: {type(reason).__name__}: {reason}; "
            "fallback=stable advanced FFmpeg path",
            flush=True,
        )
        print("[D2S_STATUS] Vulkan unavailable; using stable advanced FFmpeg path", flush=True)
        self._stop_native()
        self._close_secondary_output(self._opengl_pynv_fallback)
        self._opengl_pynv_fallback = None
        self._close_secondary_output(self._opengl_amd_fallback)
        self._opengl_amd_fallback = None
        if self._opengl_fallback is not None:
            self._opengl_fallback.close()
            self._opengl_fallback = None
        self._opengl_fallback_active = False
        fallback = self._new_host_fallback()
        self._host_fallback = fallback
        fallback.submit_frame(runtime_sbs_to_rgb(frame))

    def submit_frame(self, frame: np.ndarray) -> None:
        """Route CPU/non-CUDA frames through the same OpenGL fallback boundary.

        The native Vulkan bridge currently accepts CUDA device frames only. A
        CPU frame must not enter the Vulkan rawvideo command; probing OpenGL
        first keeps the platform decision and diagnostics identical to the
        CUDA path, then selects the stable host encoder when no GPU interop is
        available.
        """
        fallback = getattr(self, "_host_fallback", None)
        if fallback is not None:
            fallback.submit_frame(frame)
            return
        self._fallback_to_opengl(
            frame,
            RuntimeError("native Vulkan image path requires a CUDA device frame"),
        )

    def submit_cuda_frame(self, frame: Any) -> None:
        if (
            getattr(self, "_vendor_gpu_first", False)
            and not self._opengl_fallback_attempted
            and not self._native_active
            and getattr(self, "_host_fallback", None) is None
        ):
            if self._fallback_to_opengl(
                frame,
                RuntimeError("Auto vendor-native GPU stage"),
                vendor_first=True,
            ):
                return

        opengl = self._opengl_fallback
        pynv_fallback = self._opengl_pynv_fallback
        amd_fallback = self._opengl_amd_fallback
        fallback = getattr(self, "_host_fallback", None)
        if self._opengl_fallback_active and opengl is not None:
            try:
                if pynv_fallback is not None:
                    assert pynv_fallback._pynv_output is not None
                    if self._opengl_nvenc_mode == "direct":
                        try:
                            pynv_fallback._pynv_output.submit_cuda_frame(
                                opengl.submit_cuda_tensor_surface(frame)
                            )
                        except Exception as direct_exc:
                            self._opengl_nvenc_mode = "array-copy"
                            print(
                                "[OpenGLStream] direct CUDA surface write failed: "
                                f"{type(direct_exc).__name__}: {direct_exc}; "
                                "downgrade=CUDAARRAY device copy "
                                "zero_copy=False gpu_copy_count=1",
                                flush=True,
                            )
                            pynv_fallback._pynv_output.submit_cuda_frame(
                                opengl.submit_cuda_array(frame)
                            )
                    elif self._opengl_nvenc_mode == "array-copy":
                        pynv_fallback._pynv_output.submit_cuda_frame(
                            opengl.submit_cuda_array(frame)
                        )
                    else:
                        pynv_fallback._pynv_output.submit_cuda_frame(
                            opengl.submit_cuda(frame)
                        )
                elif amd_fallback is not None:
                    amd_fallback._submit_amd_packet(opengl.submit_cuda(frame))
                elif fallback is not None:
                    # The no-interop branch is intentionally host-upload. Do
                    # not perform a redundant CPU→OpenGL→CPU round trip.
                    fallback.submit_frame(runtime_sbs_to_rgb(frame))
                else:
                    raise RuntimeError("OpenGL fallback output is unavailable")
            except Exception as exc:
                self._opengl_fallback_active = False
                self._fallback_to_host(frame, exc)
            return
        if fallback is not None:
            fallback.submit_frame(runtime_sbs_to_rgb(frame))
            return
        try:
            image = self._rgba_tensor(frame)
            height, width = int(image.shape[0]), int(image.shape[1])
            if not self._native_active:
                self._start_native(width, height, cuda_device=image.device)
            if self._native_mux_process is None or self._native_mux_process.stdin is None:
                raise RuntimeError("Vulkan packet muxer stdin is unavailable")
            rgba_frame = self._native_vulkan_encoder.acquire_rgba_frame()
            ready_value = self._native_vulkan_importer.write_ffmpeg_rgba_frame(image, rgba_frame)
            timestamp = self._native_pts
            self._native_pts += 1
            self._native_vulkan_encoder.encode_rgba_frame(
                ready_value=ready_value,
                timestamp=timestamp,
            )
            while True:
                packet = self._native_vulkan_encoder.read_packet()
                if not packet:
                    break
                self._native_mux_process.stdin.write(packet)
                self._native_mux_process.stdin.flush()
        except Exception as exc:
            self._fallback_to_opengl(frame, exc)

    def close(self) -> None:
        self._stop_native()
        self._close_secondary_output(self._opengl_pynv_fallback)
        self._opengl_pynv_fallback = None
        self._close_secondary_output(self._opengl_amd_fallback)
        self._opengl_amd_fallback = None
        if self._opengl_fallback is not None:
            self._opengl_fallback.close()
            self._opengl_fallback = None
        self._opengl_fallback_active = False
        self._opengl_nvenc_mode = "none"
        fallback = getattr(self, "_host_fallback", None)
        if fallback is not None:
            self._close_secondary_output(fallback)
            self._host_fallback = None
        super().close()




class IntelD3D11DirectSbsOutput(IntelQsvDirectSbsOutput):
    """Prefer final-SBS D3D11 surface + oneVPL, then fall back to QSV.

    The native branch is opt-in until oneVPL and the D3D11 surface bridge are
    installed. It consumes the completed SBS frame, never the mono capture or
    depth input. The current bridge has one CPU upload boundary and therefore
    reports zero_copy=False.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._native_surface = None
        self._native_onevpl_encoder = None
        self._native_onevpl_mux: subprocess.Popen | None = None
        self._native_onevpl_pts = 0
        self._native_onevpl_active = False
        self._vulkan_sbs_composer = None
        self._vulkan_runtime_bridge = None

    @staticmethod
    def _rgb_to_bgra(frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected final SBS RGB8 HWC frame, got {image.shape!r}")
        bgra = np.empty((*image.shape[:2], 4), dtype=np.uint8)
        bgra[..., 0] = image[..., 2]
        bgra[..., 1] = image[..., 1]
        bgra[..., 2] = image[..., 0]
        bgra[..., 3] = 255
        return bgra

    def _native_onevpl_output_url(self) -> str:
        if self.protocol == "WEBRTC":
            return f"rtsp://127.0.0.1:{self.publish_rtsp_port}/{self.stream_key}?pkt_size=1452"
        return f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}&pkt_size=1316"

    def _start_native_onevpl_mux(self) -> None:
        command = [
            str(self.ffmpeg_path), "-hide_banner", "-loglevel", "warning",
            "-fflags", "nobuffer", "-flags", "low_delay", "-f", "h264",
            "-r", str(self.fps), "-i", "pipe:0", "-map", "0:v:0", "-c:v", "copy",
            "-muxdelay", "0", "-muxpreload", "0", "-flush_packets", "1",
            "-max_interleave_delta", "100000",
        ]
        if self.protocol == "WEBRTC":
            command.extend(["-f", "rtsp", "-rtsp_transport", "tcp", "-pkt_size", "1452"])
        else:
            command.extend(["-f", "mpegts", "-mpegts_flags", "+resend_headers"])
        command.append(self._native_onevpl_output_url())
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if self.os_name == "Windows" else 0
        self._native_onevpl_mux = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, creationflags=creationflags,
        )
        self._ffmpeg_stderr_tail = []
        self._ffmpeg_log_thread = threading.Thread(
            target=self._drain_ffmpeg_stderr,
            args=(self._native_onevpl_mux,), name="DirectSbsIntelOneVplMuxLog", daemon=True,
        )
        self._ffmpeg_log_thread.start()
        time.sleep(0.05)
        if self._native_onevpl_mux.poll() is not None:
            detail = "; ".join(self._ffmpeg_stderr_tail[-3:]) or "no FFmpeg diagnostic"
            raise RuntimeError(
                f"Intel oneVPL packet muxer exited with code "
                f"{self._native_onevpl_mux.returncode}: {detail}"
            )

    def _start_native_onevpl(self, width: int, height: int) -> None:
        if self.stereo_mix_device:
            raise RuntimeError("native Intel oneVPL final-SBS path currently requires audio disabled")
        from desktop2stereo.stereo_runtime.providers.intel.d3d11_sbs_surface import D3D11SbsSurface
        from desktop2stereo.stereo_runtime.providers.intel.onevpl_d3d11_encoder import OneVPLD3D11SurfaceEncoder

        adapter_index = int(os.environ.get("D2S_ONEVPL_D3D11_ADAPTER", "-1"))
        self._native_surface = D3D11SbsSurface(
            width=width, height=height, adapter_index=adapter_index
        )
        budget = self._dynamic_stream_rate_budget(width, height)
        target_mbps = int(budget[0] if budget else 10)
        self._native_onevpl_encoder = OneVPLD3D11SurfaceEncoder(
            width=width, height=height, fps=self.fps,
            bitrate=target_mbps * 1_000_000,
            d3d11_device=self._native_surface.device, hevc=self.use_hevc,
        )
        surface_luid = int(self._native_surface.adapter_luid)
        encoder_luid = int(self._native_onevpl_encoder.adapter_luid)
        if not surface_luid or not encoder_luid or surface_luid != encoder_luid:
            raise RuntimeError(
                "Intel D3D11 surface and oneVPL encoder Adapter LUID mismatch: "
                f"surface={surface_luid} encoder={encoder_luid}"
            )
        self._start_native_onevpl_mux()
        self._native_onevpl_pts = 0
        self._native_onevpl_active = True
        self._frame_size = (width, height)
        print(
            f"[IntelStream] native final SBS path active: RGB8 -> D3D11 BGRA8 "
            f"-> NV12 -> oneVPL -> MediaMTX; adapter_luid={self._native_surface.adapter_luid} "
            "encoder_adapter_luid_verified=True "
            "gpu_to_cpu=True zero_copy=False gpu_copy_count=1", flush=True,
        )

    def _stop_native_onevpl(self) -> None:
        self._native_onevpl_active = False
        if self._native_onevpl_encoder is not None:
            try: self._native_onevpl_encoder.close()
            except Exception: pass
            self._native_onevpl_encoder = None
        if self._native_surface is not None:
            try: self._native_surface.close()
            except Exception: pass
            self._native_surface = None
        if self._vulkan_sbs_composer is not None:
            try: self._vulkan_sbs_composer.close()
            except Exception: pass
            self._vulkan_sbs_composer = None
        if self._vulkan_runtime_bridge is not None:
            try: self._vulkan_runtime_bridge.close()
            except Exception: pass
            self._vulkan_runtime_bridge = None
        self._stop_process(self._native_onevpl_mux)
        self._native_onevpl_mux = None

    def _submit_native_onevpl(self, frame: np.ndarray) -> None:
        if not self._native_onevpl_active:
            self._start_native_onevpl(int(frame.shape[1]), int(frame.shape[0]))
        if self._native_surface is None or self._native_onevpl_encoder is None:
            raise RuntimeError("Intel oneVPL final-SBS resources are unavailable")
        self._native_surface.upload_bgra(self._rgb_to_bgra(frame))
        texture, width, height = self._native_surface.nv12_texture()
        if (width, height) != (self._native_surface.width, self._native_surface.height):
            raise RuntimeError("Intel NV12 surface dimensions changed")
        self._native_onevpl_encoder.submit_nv12(texture, self._native_onevpl_pts)
        self._native_onevpl_pts += 1
        while True:
            packet = self._native_onevpl_encoder.read_packet()
            if not packet: break
            if self._native_onevpl_mux is None or self._native_onevpl_mux.stdin is None:
                raise RuntimeError("Intel oneVPL packet muxer stdin is unavailable")
            self._native_onevpl_mux.stdin.write(packet)
            self._native_onevpl_mux.stdin.flush()

    def _disable_native_onevpl_for_audio(self) -> None:
        """Keep audio enabled by falling back from video-only oneVPL output."""
        if self.stereo_mix_device:
            print(
                "[IntelStream] native oneVPL surface path is video-only while "
                "audio is enabled; falling back to shared Intel QSV/FFmpeg "
                "audio+video path",
                flush=True,
            )
            os.environ["D2S_ONEVPL_FINAL_SBS"] = "0"
            os.environ["D2S_INTEL_VULKAN_SBS"] = "0"
            self._stop_native_onevpl()

    def submit_native_d3d11_surface(self, frame: Any) -> bool:
        """Encode a same-device final BGRA8 texture without CPU upload."""
        if self.stereo_mix_device:
            self._disable_native_onevpl_for_audio()
            return False
        enabled = os.environ.get("D2S_ONEVPL_FINAL_SBS", "0").strip().casefold()
        if enabled not in {"1", "true", "yes", "on"}:
            raise RuntimeError("native final SBS surface path is disabled")
        texture_value = getattr(frame, "texture", 0)
        device_value = getattr(frame, "device", 0)
        texture = int(getattr(texture_value, "value", texture_value) or 0)
        device = int(getattr(device_value, "value", device_value) or 0)
        producer_ready = bool(getattr(frame, "producer_ready", False))
        ready_timeline = int(getattr(frame, "ready_timeline", 0) or 0)
        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        adapter_luid = int(getattr(frame, "adapter_luid", 0) or 0)
        if not texture or not device or not width or not height or not adapter_luid:
            raise RuntimeError("native final SBS frame lacks D3D11 texture contract")
        if hasattr(frame, "producer_ready") and not producer_ready:
            raise RuntimeError("native final SBS frame is not producer-ready")
        if hasattr(frame, "ready_timeline") and ready_timeline < 1:
            raise RuntimeError("native final SBS frame lacks completed Vulkan producer timeline")
        from desktop2stereo.stereo_runtime.providers.intel.d3d11_sbs_surface import D3D11SbsSurface
        from desktop2stereo.stereo_runtime.providers.intel.onevpl_d3d11_encoder import OneVPLD3D11SurfaceEncoder

        if self._native_surface is None:
            self._native_surface = D3D11SbsSurface(
                width=width, height=height, d3d11_device=device
            )
            if self._native_surface.adapter_luid != adapter_luid:
                raise RuntimeError(
                    "native final SBS frame Adapter LUID does not match D3D11 surface"
                )
            budget = self._dynamic_stream_rate_budget(width, height)
            target_mbps = int(budget[0] if budget else 10)
            self._native_onevpl_encoder = OneVPLD3D11SurfaceEncoder(
                width=width, height=height, fps=self.fps,
                bitrate=target_mbps * 1_000_000,
                d3d11_device=self._native_surface.device, hevc=self.use_hevc,
            )
            if self._native_onevpl_encoder.adapter_luid != adapter_luid:
                raise RuntimeError("native final SBS encoder Adapter LUID mismatch")
            self._start_native_onevpl_mux()
            self._native_onevpl_active = True
            self._native_onevpl_pts = 0
            self._frame_size = (width, height)
            print(
                "[IntelStream] native final SBS texture import active: "
                "D3D11 BGRA8 -> VideoProcessor NV12 -> oneVPL -> MediaMTX; "
                f"adapter_luid={adapter_luid} gpu_to_cpu=False "
                "zero_copy=False gpu_copy_count=1 "
                "zero_copy_gate=unverified_intel_target",
                flush=True,
            )
        elif (
            self._native_surface.width != width
            or self._native_surface.height != height
            or self._native_surface.adapter_luid != adapter_luid
        ):
            raise RuntimeError("native final SBS texture contract changed during stream")
        if self._native_surface is None or self._native_onevpl_encoder is None:
            raise RuntimeError("Intel native final SBS texture resources are unavailable")
        self._native_surface.set_bgra_texture(texture, adapter_luid=adapter_luid)
        _texture, out_width, out_height = self._native_surface.nv12_texture()
        if (out_width, out_height) != (width, height):
            raise RuntimeError("Intel imported final SBS surface dimensions changed")
        self._native_onevpl_encoder.submit_nv12(_texture, self._native_onevpl_pts)
        self._native_onevpl_pts += 1
        while True:
            packet = self._native_onevpl_encoder.read_packet()
            if not packet:
                break
            if self._native_onevpl_mux is None or self._native_onevpl_mux.stdin is None:
                raise RuntimeError("Intel oneVPL packet muxer stdin is unavailable")
            self._native_onevpl_mux.stdin.write(packet)
            self._native_onevpl_mux.stdin.flush()
        return True

    def submit_vulkan_stereo_frame(self, runtime_result: Any) -> bool:
        """Compose Vulkan eye resources into the Intel D3D11 encoder surface."""
        if self.stereo_mix_device:
            self._disable_native_onevpl_for_audio()
            return False
        enabled = os.environ.get("D2S_ONEVPL_FINAL_SBS", "0").strip().casefold()
        if enabled not in {"1", "true", "yes", "on"}:
            raise RuntimeError("native final SBS surface path is disabled")
        deferred_request = getattr(runtime_result, "vulkan_compute_request", None)
        if deferred_request is not None:
            from desktop2stereo.stereo_runtime.intel_vulkan_sbs import (
                IntelVulkanSbsRuntimeBridge,
            )
            shape = tuple(int(value) for value in getattr(deferred_request.rgb, "shape", ()))
            if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
                raise RuntimeError(f"deferred Vulkan SBS request has invalid RGB shape: {shape}")
            height, width = shape[-2:]
            try:
                if self._vulkan_runtime_bridge is None:
                    self._vulkan_runtime_bridge = IntelVulkanSbsRuntimeBridge(width, height)
                elif (
                    self._vulkan_runtime_bridge.eye_width != width
                    or self._vulkan_runtime_bridge.eye_height != height
                ):
                    raise RuntimeError("deferred Vulkan SBS dimensions changed during stream")
                frame = self._vulkan_runtime_bridge.submit(deferred_request)
                self.submit_native_d3d11_surface(frame)
            except Exception as exc:
                print(
                    f"[IntelStream] Vulkan deferred SBS unavailable: {exc}; "
                    "falling back to the regular Intel QSV/D3D11 path",
                    flush=True,
                )
                os.environ["D2S_INTEL_VULKAN_SBS"] = "0"
                self._stop_native_onevpl()
                return False
            return True
        left_eye = getattr(runtime_result, "left_eye", None)
        right_eye = getattr(runtime_result, "right_eye", None)
        context = getattr(left_eye, "context", None)
        if context is None or getattr(right_eye, "context", None) is not context:
            raise RuntimeError("Vulkan stereo frame lacks a shared Vulkan context")
        width = int(getattr(left_eye, "width", 0) or 0)
        height = int(getattr(left_eye, "height", 0) or 0)
        if not width or not height:
            raise RuntimeError("Vulkan stereo frame dimensions are unavailable")
        if self._vulkan_sbs_composer is None:
            from desktop2stereo.stereo_runtime.intel_vulkan_sbs import IntelVulkanSbsComposer
            from desktop2stereo.stereo_runtime.providers.intel.onevpl_d3d11_encoder import (
                OneVPLD3D11SurfaceEncoder,
            )

            self._vulkan_sbs_composer = IntelVulkanSbsComposer(context, width, height)
            self._native_surface = self._vulkan_sbs_composer.surface
            budget = self._dynamic_stream_rate_budget(width * 2, height)
            target_mbps = int(budget[0] if budget else 10)
            self._native_onevpl_encoder = OneVPLD3D11SurfaceEncoder(
                width=width * 2,
                height=height,
                fps=self.fps,
                bitrate=target_mbps * 1_000_000,
                d3d11_device=self._native_surface.device,
                hevc=self.use_hevc,
            )
            if self._native_onevpl_encoder.adapter_luid != self._native_surface.adapter_luid:
                raise RuntimeError("Intel Vulkan/D3D11/oneVPL Adapter LUID mismatch")
            self._start_native_onevpl_mux()
            self._native_onevpl_active = True
            self._native_onevpl_pts = 0
            self._frame_size = (width * 2, height)
            print(
                "[IntelStream] Vulkan eyes -> D3D11 shared BGRA SBS -> "
                "VideoProcessor NV12 -> oneVPL -> MediaMTX; gpu_to_cpu=False "
                "zero_copy=False gpu_copy_count=1 "
                "(Vulkan producer wait verified; end-to-end zero-copy remains gated)",
                flush=True,
            )
        elif (
            self._vulkan_sbs_composer.eye_width != width
            or self._vulkan_sbs_composer.eye_height != height
        ):
            raise RuntimeError("Vulkan SBS dimensions changed during stream")
        ready_timeline = int(
            (getattr(runtime_result, "debug_info", None) or {}).get(
                "vulkan_submit_timeline", 0
            )
            or 0
        )
        frame = self._vulkan_sbs_composer.compose(
            left_eye,
            right_eye,
            ready_timeline=ready_timeline or None,
        )
        self.submit_native_d3d11_surface(frame)
        return True

    def submit_frame(self, frame: np.ndarray) -> None:
        enabled = os.environ.get("D2S_ONEVPL_FINAL_SBS", "0").strip().casefold()
        if enabled not in {"1", "true", "yes", "on"}:
            super().submit_frame(frame)
            return
        try:
            self._submit_native_onevpl(frame)
        except Exception as exc:
            print(
                f"[IntelStream] native oneVPL final-SBS path unavailable: {exc}; "
                "falling back to Intel QSV/D3D11 FFmpeg path", flush=True,
            )
            self._stop_native_onevpl()
            super().submit_frame(frame)

    def close(self) -> None:
        self._stop_native_onevpl()
        super().close()


class PyNvDirectSbsOutput(_PyNvDirectSbsOutputMixin, FfmpegDirectSbsOutput):
    pass


class _AmdAmfDirectSbsOutputMixin:
    """Submit ROCm tensors to the native HIP→D3D11→AMF bridge."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._amd_encoder = None
        self._amd_packet_process: subprocess.Popen | None = None
        self._amd_fallback: FfmpegDirectSbsOutput | None = None

    @staticmethod
    def _hip_rgba_tensor(frame):
        import torch

        image = getattr(frame, "sbs", frame)
        if not isinstance(image, torch.Tensor) or not bool(getattr(image, "is_cuda", False)):
            raise RuntimeError("AMD AMF requires a ROCm device tensor")
        if image.ndim == 4:
            if int(image.shape[0]) != 1:
                raise RuntimeError("AMD AMF accepts one SBS frame at a time")
            image = image[0]
        if image.ndim != 3:
            raise RuntimeError(f"unsupported AMD AMF tensor shape: {tuple(image.shape)!r}")
        if int(image.shape[0]) in (3, 4):
            image = image.permute(1, 2, 0)
        if int(image.shape[-1]) not in (3, 4):
            raise RuntimeError(f"AMD AMF requires RGB/RGBA tensor: {tuple(image.shape)!r}")
        if image.dtype != torch.uint8:
            image = image.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        if int(image.shape[-1]) == 3:
            alpha = torch.full(
                (*image.shape[:2], 1), 255, dtype=torch.uint8, device=image.device
            )
            image = torch.cat((image, alpha), dim=-1)
        return image.contiguous()

    def _start_amd_encoder(self, image) -> None:
        if self.stereo_mix_device:
            raise RuntimeError("native AMD AMF path requires audio to be disabled")
        from streaming.amd_encoder import AmdAmfSurfaceEncoder

        height, width = int(image.shape[0]), int(image.shape[1])
        budget = self._dynamic_stream_rate_budget(width, height)
        bitrate = int((budget[0] if budget is not None else 10) * 1_000_000)
        self._amd_encoder = AmdAmfSurfaceEncoder(
            width, height, self.fps, bitrate, hevc=self.use_hevc
        )
        codec = "hevc" if self.use_hevc else "h264"
        destination = (
            f"srt://127.0.0.1:8890?streamid=publish:{self.stream_key}&pkt_size=1316"
        )
        self._amd_packet_process = subprocess.Popen(
            [
                str(self.ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "warning",
                "-fflags",
                "nobuffer",
                "-f",
                codec,
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "copy",
                "-f",
                "mpegts",
                destination,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if self.os_name == "Windows"
                else 0
            ),
        )
        self._frame_size = (width, height)
        print(
            f"[DirectSbsStream] AMD HIP→D3D11→AMF GPU path active: "
            f"{width}x{height}@{self.fps}",
            flush=True,
        )

    def _submit_amd_packet(self, image) -> None:
        import torch

        if self._amd_encoder is None or self._amd_packet_process is None:
            self._start_amd_encoder(image)
        assert self._amd_encoder is not None
        process = self._amd_packet_process
        if process.poll() is not None or process.stdin is None:
            raise RuntimeError("AMF packet muxer is unavailable")
        stream = int(torch.cuda.current_stream(device=image.device).cuda_stream)
        self._amd_encoder.submit_hip_rgba(
            int(image.data_ptr()), int(image.stride(0) * image.element_size()), stream
        )
        while True:
            packet = self._amd_encoder.read_packet()
            if not packet:
                break
            process.stdin.write(packet)
            process.stdin.flush()

    def _fallback_to_ffmpeg(self, frame, reason: Exception) -> None:
        print(
            f"[DirectSbsStream] AMD native GPU path unavailable: {reason}; "
            "falling back to FFmpeg hardware/software encoding",
            flush=True,
        )
        if self._amd_packet_process is not None:
            self._stop_process(self._amd_packet_process)
            self._amd_packet_process = None
        if self._amd_encoder is not None:
            self._amd_encoder.close()
            self._amd_encoder = None
        self._amd_fallback = FfmpegDirectSbsOutput(
            base_dir=self.base_dir,
            protocol=self.protocol,
            port=self.port,
            stream_key=self.stream_key,
            fps=self.fps,
            crf=self.crf,
            stereo_mix_device=self.stereo_mix_device,
            audio_delay=self.audio_delay,
            os_name=self.os_name,
            prefer_nvenc=self.prefer_nvenc,
            display_mode=self.display_mode,
        )
        self._amd_fallback.server_process = self.server_process
        self._amd_fallback.submit_frame(runtime_sbs_to_rgb(frame))

    def submit_cuda_frame(self, frame: Any) -> None:
        if self._amd_fallback is not None:
            self._amd_fallback.submit_frame(runtime_sbs_to_rgb(frame))
            return
        try:
            self._submit_amd_packet(self._hip_rgba_tensor(frame))
        except Exception as exc:
            self._fallback_to_ffmpeg(frame, exc)

    def close(self) -> None:
        if self._amd_packet_process is not None:
            self._stop_process(self._amd_packet_process)
            self._amd_packet_process = None
        if self._amd_encoder is not None:
            self._amd_encoder.close()
            self._amd_encoder = None
        if self._amd_fallback is not None:
            self._amd_fallback.close()
            self._amd_fallback = None
        super().close()


class AmdAmfDirectSbsOutput(_AmdAmfDirectSbsOutputMixin, FfmpegDirectSbsOutput):
    pass
