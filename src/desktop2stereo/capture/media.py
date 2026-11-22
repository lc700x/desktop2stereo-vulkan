from __future__ import annotations

import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .media_browser import VIDEO_EXTENSIONS, MediaBrowser, media_kind_for_path
from .media_cadence import SourceFrameSchedule, display_aligned_media_fps

_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
_VIDEO_EXTENSIONS = VIDEO_EXTENSIONS
_NVDEC_DLL_DIRECTORY_HANDLES = []
_MEDIA_CADENCE_TARGET_LOCK = threading.Lock()
_MEDIA_CADENCE_TARGET_FPS: float | None = None
_AMF_DECODERS = {
    "h264": "h264_amf",
    "hevc": "hevc_amf",
    "av1": "av1_amf",
    "vp9": "vp9_amf",
}


class _HardwareDecodeUnavailable(RuntimeError):
    pass


def set_media_playback_cadence_target(fps: float | None) -> None:
    """Publish a process-local cadence target for the active media source."""
    global _MEDIA_CADENCE_TARGET_FPS
    try:
        value = float(fps) if fps is not None else None
    except (TypeError, ValueError):
        value = None
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        value = None
    with _MEDIA_CADENCE_TARGET_LOCK:
        _MEDIA_CADENCE_TARGET_FPS = value


def get_media_playback_cadence_target() -> float | None:
    with _MEDIA_CADENCE_TARGET_LOCK:
        return _MEDIA_CADENCE_TARGET_FPS


def reset_media_playback_cadence_target() -> None:
    set_media_playback_cadence_target(None)


def _is_hardware_decode_unavailable(exc: Exception) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError, _HardwareDecodeUnavailable)):
        return True
    code = getattr(exc, "error_code", getattr(exc, "code", None))
    try:
        if int(code) == 801:
            return True
    except (TypeError, ValueError):
        pass
    name = type(exc).__name__.casefold()
    message = str(exc).casefold()
    return "unsupported" in name or any(
        marker in message
        for marker in (
            "not supported on this gpu",
            "not supported on this device",
            "codec not supported",
            "unsupported codec",
            "profile not supported",
            "profile is not supported",
            "bit depth not supported",
            "chroma format not supported",
            "hardware decoder unavailable",
            "nvdec is unavailable",
            "no nvdec",
            "unsupported device_type",
            "no amf capable device",
            "amf decoder is not supported",
            "amf decoder unsupported",
            "amf failed to initialize",
            "failed to initialize amf",
            "platform doesn't support hardware accelerated",
        )
    )


def _is_software_decoder_unavailable(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "decoder not found",
            "no decoder found",
            "no suitable decoder",
            "decoder unavailable",
            "unknown decoder",
        )
    )


def _preferred_audio_output_device(sounddevice_module) -> int | None:
    """Prefer WASAPI on Windows; MME can report no driver during playback writes."""
    override = os.environ.get("D2S_MEDIA_AUDIO_DEVICE", "").strip()
    if override:
        try:
            device = int(override)
        except ValueError:
            return None
        return device if device >= 0 else None
    if os.name != "nt":
        return None
    try:
        hostapis = sounddevice_module.query_hostapis()
    except Exception:
        return None
    for hostapi in hostapis:
        name = str(hostapi.get("name") or "").casefold()
        if "wasapi" not in name:
            continue
        try:
            device = int(hostapi.get("default_output_device", -1))
        except (TypeError, ValueError):
            return None
        return device if device >= 0 else None
    return None


def _is_amf_vpp_unavailable(message: str) -> bool:
    text = str(message).casefold()
    return "vpp_amf" in text and any(
        marker in text
        for marker in (
            "no such filter",
            "not found",
            "unsupported",
            "not supported",
            "failed to initialize",
            "error initializing",
            "failed to configure",
            "could not configure",
            "impossible to convert between the formats",
            "could not convert between the formats",
        )
    )


def _enable_torch_cuda_dlls() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    torch_lib = Path(__file__).resolve().parents[2] / "python3_cuda" / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.is_dir():
        _NVDEC_DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(torch_lib)))


def _media_gpu_vendor() -> str:
    """Return the active torch accelerator vendor without probing devices."""
    try:
        import torch
    except Exception:
        return "unknown"
    if getattr(getattr(torch, "version", None), "hip", None):
        return "amd"
    if getattr(getattr(torch, "version", None), "cuda", None):
        try:
            name = str(torch.cuda.get_device_name(0) or "").casefold()
        except Exception:
            name = ""
        return "amd" if "amd" in name or "radeon" in name else "nvidia"
    return "unknown"


def _resolve_media_playback_cadence(
    source_fps: float,
    width: int,
    height: int,
    requested_fps: float,
    *,
    target_fps: float | None = None,
) -> tuple[float, int]:
    """Select a source divisor under the requested/measured ceiling, never slow time."""
    source = max(1.0, float(source_fps or 0.0))
    ceiling = min(source, max(1.0, float(requested_fps or source)))
    for target in (get_media_playback_cadence_target(), target_fps):
        if target is not None and math.isfinite(target) and target > 0.0:
            ceiling = min(ceiling, target)
    try:
        override = float(os.environ.get("D2S_MEDIA_DECIMATION_TARGET_FPS", "0") or 0.0)
    except ValueError:
        override = 0.0
    if math.isfinite(override) and override > 0.0:
        ceiling = min(ceiling, override)
    step = max(1, math.ceil(source / ceiling - 1e-6))
    return source / step, step


def _select_media_cadence_fps_for_display(
    source_fps: float,
    display_fps: float,
    max_fps: float,
    *,
    min_fps: float = 12.0,
) -> float | None:
    """Choose the highest source division that lands on integral display ticks."""
    return display_aligned_media_fps(
        float(source_fps or 0.0),
        float(display_fps or 0.0),
        float(max_fps or 0.0),
        min_fps=float(min_fps),
    )


def _resolve_media_mode(path: str, configured_mode: str) -> str:
    mode = str(configured_mode or "").strip().casefold()
    if mode in {"video", "image"}:
        return mode
    kind = media_kind_for_path(path) if path else ""
    return "image" if kind == "image" else "video"


def _browser_output_size(config, width: int, height: int) -> tuple[int, int]:
    if width > 0 and height > 0:
        output = (int(width), int(height))
    else:
        resolution = getattr(config, "output_resolution", 1080)
        if isinstance(resolution, (tuple, list)) and len(resolution) >= 2:
            output = (max(2, int(resolution[0])), max(2, int(resolution[1])))
        else:
            output = (1920, 1080)
    # The browser panel is an overlay; a 720p neutral backdrop is sufficient
    # and avoids running depth/stereo work at 4K while the user is browsing.
    scale = min(1.0, 1280.0 / max(output))
    return (
        max(2, int(round(output[0] * scale)) // 2 * 2),
        max(2, int(round(output[1] * scale)) // 2 * 2),
    )


def _mount_iso(path: Path) -> tuple[Path, bool]:
    if os.name != "nt":
        raise ValueError("Mount the ISO with the operating system, then select its mounted disc folder")
    env = os.environ.copy()
    env["D2S_MEDIA_ISO_PATH"] = str(path)
    command = (
        "$ErrorActionPreference='Stop'; "
        "$p=$env:D2S_MEDIA_ISO_PATH; "
        "$image=Get-DiskImage -ImagePath $p -ErrorAction SilentlyContinue; "
        "$owned=$false; "
        "if(-not $image -or -not $image.Attached){"
        "$image=Mount-DiskImage -ImagePath $p -ReadOnly -PassThru; $owned=$true}; "
        "$volume=$image | Get-Disk | Get-Partition | Get-Volume | "
        "Where-Object DriveLetter | Select-Object -First 1; "
        "if(-not $volume){throw 'The ISO has no mounted volume'}; "
        "[pscustomobject]@{drive=$volume.DriveLetter;owned=$owned} | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    mounted = json.loads(result.stdout)
    drive = str(mounted.get("drive") or "").strip()
    if not drive:
        raise RuntimeError("Windows mounted the ISO without assigning a drive letter")
    return Path(f"{drive}:\\"), bool(mounted.get("owned"))


def _dismount_iso(path: str) -> None:
    env = os.environ.copy()
    env["D2S_MEDIA_ISO_PATH"] = path
    subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
            "Dismount-DiskImage -ImagePath $env:D2S_MEDIA_ISO_PATH -ErrorAction Stop",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _parse_bluray_playlist(path: Path, stream_dir: Path) -> list[tuple[Path, float, float]]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"MPLS":
        return []
    playlist_offset = int.from_bytes(data[8:12], "big")
    if playlist_offset + 10 > len(data):
        return []
    playlist_length = int.from_bytes(data[playlist_offset:playlist_offset + 4], "big")
    playlist_end = playlist_offset + 4 + playlist_length
    item_count = int.from_bytes(data[playlist_offset + 6:playlist_offset + 8], "big")
    if playlist_end > len(data) or item_count > 10000:
        return []
    cursor = playlist_offset + 10
    entries = []
    for _ in range(item_count):
        if cursor + 2 > playlist_end:
            return []
        item_length = int.from_bytes(data[cursor:cursor + 2], "big")
        item_start = cursor + 2
        item_end = item_start + item_length
        if item_length < 20 or item_end > playlist_end:
            return []
        item = data[item_start:item_end]
        try:
            clip_name = item[:5].decode("ascii")
        except UnicodeDecodeError:
            return []
        clip = stream_dir / f"{clip_name}.m2ts"
        inpoint = int.from_bytes(item[12:16], "big") / 45000.0
        outpoint = int.from_bytes(item[16:20], "big") / 45000.0
        if not clip.is_file() or outpoint <= inpoint:
            return []
        entries.append((clip, inpoint, outpoint))
        cursor = item_end
    return entries


def _disc_entries(root: Path) -> list[tuple[Path, float | None, float | None]]:
    if not root.is_dir():
        return []
    bluray_root = root if root.name.casefold() == "bdmv" else root / "BDMV"
    stream_dir = bluray_root / "STREAM"
    if stream_dir.is_dir():
        playlists = [bluray_root / "PLAYLIST", bluray_root / "BACKUP" / "PLAYLIST"]
        candidates = []
        for playlist_dir in playlists:
            if not playlist_dir.is_dir():
                continue
            for playlist in playlist_dir.glob("*.mpls"):
                if playlist.stat().st_size > 1024 * 1024:
                    continue
                entries = _parse_bluray_playlist(playlist, stream_dir)
                if entries:
                    duration = sum(end - start for _, start, end in entries)
                    size = sum(clip.stat().st_size for clip, _, _ in entries)
                    candidates.append((duration, size, entries))
        if candidates:
            _, _, entries = max(candidates, key=lambda item: (item[0], item[1]))
            return entries
        clips = [path for path in stream_dir.glob("*.m2ts") if path.is_file()]
        if clips:
            return [(max(clips, key=lambda path: path.stat().st_size), None, None)]

    video_ts = root if root.name.casefold() == "video_ts" else root / "VIDEO_TS"
    if video_ts.is_dir():
        titles: dict[str, list[tuple[int, Path]]] = {}
        for path in video_ts.glob("VTS_*.VOB"):
            parts = path.stem.upper().split("_")
            if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
                continue
            title, segment = parts[1], int(parts[2])
            if segment > 0 and path.is_file():
                titles.setdefault(title, []).append((segment, path))
        if titles:
            segments = max(
                titles.values(),
                key=lambda items: sum(path.stat().st_size for _, path in items),
            )
            return [(path, None, None) for _, path in sorted(segments)]

    files = [path for path in root.iterdir() if path.is_file() and path.suffix.casefold() in _VIDEO_EXTENSIONS]
    if files:
        return [(max(files, key=lambda path: path.stat().st_size), None, None)]
    return []


def _write_concat_manifest(entries: list[tuple[Path, float | None, float | None]]) -> str:
    descriptor, manifest_path = tempfile.mkstemp(prefix="d2s-media-", suffix=".ffconcat")
    lines = ["ffconcat version 1.0"]
    for path, inpoint, outpoint in entries:
        escaped = str(path.resolve()).replace("\\", "/").replace("'", r"'\''")
        lines.append(f"file '{escaped}'")
        if inpoint is not None:
            lines.append(f"inpoint {max(0.0, float(inpoint)):.6f}")
        if outpoint is not None:
            lines.append(f"outpoint {max(0.0, float(outpoint)):.6f}")
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    return manifest_path


def _ffmpeg_paths() -> tuple[str, str]:
    configured = os.environ.get("D2S_FFMPEG_PATH", "").strip()
    candidates = [Path(configured)] if configured else []
    runtime_root = Path(
        os.environ.get(
            "D2S_STREAMING_RUNTIME_DIR",
            Path(__file__).resolve().parents[1] / "streaming" / "rtmp",
        )
    )
    executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates.append(runtime_root / "ffmpeg" / "bin" / executable)
    ffmpeg = next((path for path in candidates if path.is_file()), None)
    ffmpeg_text = str(ffmpeg) if ffmpeg is not None else shutil.which("ffmpeg")
    if not ffmpeg_text:
        raise FileNotFoundError("FFmpeg was not found in the application runtime or PATH")
    ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    sibling_probe = Path(ffmpeg_text).with_name(ffprobe_name)
    ffprobe = str(sibling_probe) if sibling_probe.is_file() else shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("FFprobe was not found next to FFmpeg or in PATH")
    return ffmpeg_text, ffprobe


def _probe_media(ffprobe: str, input_args: list[str]) -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,duration,color_transfer,color_primaries,color_space,color_range,chroma_location,pix_fmt",
            "-show_entries", "format=duration",
            "-of", "json",
            *input_args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("The selected file has no video stream")
    duration = video.get("duration") or (data.get("format") or {}).get("duration") or 0
    try:
        duration = max(0.0, float(duration))
    except (TypeError, ValueError):
        duration = 0.0
    fps = 0.0
    try:
        numerator, denominator = str(video.get("avg_frame_rate", "0/1")).split("/", 1)
        fps = float(numerator) / max(float(denominator), 1.0)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return {
        "codec_name": str(video.get("codec_name") or "").casefold(),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "duration": duration,
        "fps": fps,
        "transfer": str(video.get("color_transfer") or "").casefold(),
        "primaries": str(video.get("color_primaries") or "").casefold(),
        "colorspace": str(video.get("color_space") or "").casefold(),
        "color_range": str(video.get("color_range") or "").casefold(),
        "chroma_location": str(video.get("chroma_location") or "left").casefold(),
        "pix_fmt": str(video.get("pix_fmt") or "").casefold(),
        "audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def _video_pixel_format(info: dict) -> str:
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    pixel_format = str(info.get("pix_fmt") or "").casefold()
    colorspace = str(info.get("colorspace") or "").casefold()
    chroma_location = str(info.get("chroma_location") or "left").casefold()
    transfer = str(info.get("transfer") or "").casefold()
    primaries = str(info.get("primaries") or "").casefold()
    supported_matrices = {"bt709", "bt2020nc", "smpte170m", "bt470bg"}
    supported_transfers = {"", "bt709", "smpte170m", "smpte2084", "arib-std-b67"}
    supported_primaries = {"", "bt709", "bt2020", "smpte432", "smpte431"}
    supported_chroma = {"left", "center", "topleft"}
    is_420 = "420" in pixel_format or pixel_format in {"nv12", "p010", "p010le"}
    if (
        width > 0
        and height > 0
        and width % 2 == 0
        and height % 2 == 0
        and is_420
        and colorspace in supported_matrices
        and transfer in supported_transfers
        and primaries in supported_primaries
        and chroma_location in supported_chroma
    ):
        return "p010le" if "10" in pixel_format or pixel_format.startswith("p010") else "nv12"
    return "bgr24"


def _media_frame_format(info: dict, pixel_format: str) -> str:
    if pixel_format in {"nv12", "p010le"}:
        transfer = str(info.get("transfer") or "").casefold()
        hdr = transfer in _HDR_TRANSFERS
        transfer_tag = (
            "pq" if transfer == "smpte2084"
            else "hlg" if transfer == "arib-std-b67"
            else "sdr"
        )
        primaries = str(info.get("primaries") or "").casefold()
        colorspace = str(info.get("colorspace") or "").casefold()
        if primaries in {"", "unknown"}:
            primaries = "bt2020" if hdr else "bt709"
        if colorspace in {"", "unknown"}:
            colorspace = "bt2020nc" if hdr else "bt709"
        color_range = "pc" if str(info.get("color_range") or "").casefold() == "pc" else "tv"
        chroma_location = str(info.get("chroma_location") or "left").casefold()
        return f"{pixel_format}_{transfer_tag}_{primaries}_{colorspace}_{color_range}_{chroma_location}"
    transfer = str(info.get("transfer") or "").casefold()
    primaries = str(info.get("primaries") or "").casefold()
    if primaries in {"", "unknown"}:
        primaries = "bt2020" if transfer in _HDR_TRANSFERS else ""
    if transfer in _HDR_TRANSFERS:
        transfer_tag = "pq" if transfer == "smpte2084" else "hlg"
        return f"bgr24_{transfer_tag}_{primaries or 'bt2020'}"
    if primaries == "bt2020":
        return "bgr24_sdr_bt2020"
    return "bgr24"


def _video_filter(info: dict, output_fps: int) -> str:
    pixel_format = _video_pixel_format(info)
    source_fps = info["fps"]
    fps = min(float(output_fps), source_fps) if source_fps > 0 else float(output_fps)
    if pixel_format == "nv12":
        # Keep subsampled YUV on the CPU pipe. The capture preprocessor uploads
        # half as many bytes and performs YUV/RGB, gamut and HDR work on GPU.
        color = "format=nv12"
    elif pixel_format == "p010le":
        # Preserve 10-bit YUV samples for the GPU HDR conversion path.
        color = "format=p010le"
    elif info["transfer"] in _HDR_TRANSFERS:
        # Preserve PQ/HLG values; the shared torch preprocess performs the
        # gamut conversion and tone mapping on the selected compute device.
        color = "format=bgr24"
    else:
        primaries = info["primaries"]
        input_profile = (
            primaries if primaries in {"bt709", "bt2020", "bt470bg", "smpte170m"}
            else "bt709"
        )
        color = (
            f"colorspace=all=bt709:iall={input_profile}:format=yuv444p12,"
            "format=bgr24"
        )
    return f"fps={max(1.0, fps):.6g},{color}"


def _resolve_amf_output_size(info: dict, render_size_config) -> tuple[int, int] | None:
    if render_size_config is None or _video_pixel_format(info) not in {"nv12", "p010le"}:
        return None
    if str(info.get("transfer") or "").casefold() in _HDR_TRANSFERS:
        # AMF VPP currently emits black P010 frames for HDR sources on the
        # tested NVIDIA and AMD stacks. Keep the decoded frame intact and let
        # the GPU preprocessor perform HDR conversion and UI-requested resize.
        return None
    source_width = int(info.get("width") or 0)
    source_height = int(info.get("height") or 0)
    if source_width <= 0 or source_height <= 0:
        return None
    from stereo_runtime.render_size import resolve_render_size

    target_width, target_height = resolve_render_size(
        (source_width, source_height), render_size_config
    )
    target_width = max(2, int(target_width) // 2 * 2)
    target_height = max(2, int(target_height) // 2 * 2)
    if target_width >= source_width or target_height >= source_height:
        return None
    return target_width, target_height


def _image_to_bgr(image) -> np.ndarray:
    """Convert a PIL image (including float/HDR stills) to a BGR frame."""
    if image.mode in {"F", "I", "I;16", "I;16L", "I;16B"}:
        values = np.asarray(image, dtype=np.float32)
        if values.size and float(np.nanmax(values)) > 255.0:
            values = values / max(float(np.nanmax(values)), 1.0)
        values = np.clip(values, 0.0, 1.0)
        gray = (values * 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
    return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def _load_still_image(path: str) -> np.ndarray:
    """Decode the broad still-image set with optional Pillow/FFmpeg backends."""
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:
        pass
    try:
        with Image.open(path) as source:
            if getattr(source, "n_frames", 1) > 1:
                source.seek(0)
            return _image_to_bgr(source)
    except Exception:
        pass
    try:
        import cv2

        decoded = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if decoded is not None:
            if decoded.dtype != np.uint8:
                values = decoded.astype(np.float32)
                peak = float(np.nanmax(values)) if values.size else 0.0
                if peak > 1.0:
                    values *= 255.0 / max(peak, 255.0)
                elif peak > 0.0:
                    values *= 255.0
                decoded = np.clip(values, 0.0, 255.0).astype(np.uint8)
            if decoded.ndim == 2:
                return cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
            if decoded.shape[2] == 4:
                return cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
            return decoded[:, :, :3]
    except Exception:
        pass
    if Path(path).suffix.casefold() in {".svg", ".svgz"}:
        try:
            import cairosvg

            png = cairosvg.svg2png(url=path)
            with Image.open(io.BytesIO(png)) as source:
                return _image_to_bgr(source)
        except Exception:
            pass
    ffmpeg, ffprobe = _ffmpeg_paths()
    info = _probe_media(ffprobe, ["-i", str(path)])
    width, height = int(info["width"]), int(info["height"])
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    expected = width * height * 3
    if len(result.stdout) < expected:
        raise ValueError(f"FFmpeg returned an incomplete image frame for: {path}")
    return np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape(
        height, width, 3
    ).copy()


def _amf_vpp_filter(
    info: dict, output_fps: int, output_size: tuple[int, int] | None
) -> str:
    if output_size is None:
        return ""
    pixel_format = _video_pixel_format(info)
    if pixel_format not in {"nv12", "p010le"}:
        return ""
    width, height = output_size
    fps_filter = _video_filter(info, output_fps).split(",", 1)[0]
    return (
        f"vpp_amf=w={width}:h={height}:format={pixel_format},"
        f"hwdownload,format={pixel_format},{fps_filter}"
    )


class MediaCaptureSource:
    """Decode one local image or video into the shared capture-frame pipeline."""

    def __init__(self, config):
        self.config = config
        self.path = str(config.media_path or "").strip()
        self._configured_mode = str(config.capture_mode or "Video").strip()
        self.mode = _resolve_media_mode(self.path, self._configured_mode)
        self.width = 0
        self.height = 0
        self.duration = 0.0
        self.position = max(0.0, float(config.media_resume_position or 0.0))
        self.volume = max(0.0, min(1.0, float(getattr(config, "media_volume", 1.0) or 1.0)))
        self.muted = bool(getattr(config, "media_muted", False))
        self.playing = self.mode == "video"
        self.hdr = False
        self.error = ""
        self.audio_error = ""
        self._browser = MediaBrowser(
            initial_directory=(
                getattr(config, "media_browser_directory", None)
                or (str(Path(self.path).parent) if self.path else None)
            ),
            history_path=getattr(config, "media_history_path", None),
        )
        self._browser_mode = False
        self._last_browser_frame_at = 0.0
        self._frame = None
        self._displayed_frame = None
        self._ffmpeg = None
        self._video_filter = ""
        self._amf_video_filter = ""
        self._amf_output_size = None
        self._decode_width = 0
        self._decode_height = 0
        self._frame_pixel_format = "bgr24"
        self._ffmpeg_frame_format = "bgr24"
        self._input_args = []
        self._nvdec_source_path = None
        self._nvdec_decoder = None
        self._nvdec_torch = None
        self._nvdec_lock = threading.RLock()
        self._nvdec_buffer_size = 4
        self._nvdec_disabled = False
        self._codec_name = ""
        self._amf_decoder = None
        self._amf_disabled = False
        self._nvdec_first_frame_pending = False
        self._nvdec_next_frame_at = None
        self._nvdec_skip_fraction = 0.0
        self._decode_backend = "ffmpeg-software"
        self.decode_fallback_reason = ""
        self._source_fps = 0.0
        self._playback_fps = max(1.0, float(config.fps))
        self.transfer = ""
        self.primaries = ""
        self._manifest_path = None
        self._mounted_iso_path = None
        self._owns_iso_mount = False
        self._media_generation = 0
        self._source_frame_index = 0
        self._current_source_frame_identity = None
        self._session_id = str(getattr(config, "media_session_id", "") or "")
        self._video_process = None
        self._opencv_capture = None
        self._opencv_next_frame_at = None
        self._opencv_skip_fraction = 0.0
        self._opencv_realtime = False
        self._native_decoder = None
        self._native_frame_importer = None
        self._native_frame_pts = None
        self._native_frame_duration = 0.0
        self._audio_process = None
        self._audio_stream = None
        self._audio_thread = None
        self._audio_started_at = None
        self._audio_position_anchor = self.position
        self._audio_stop = threading.Event()
        self._audio_sync = None
        self._presented_media_identity = None
        self._presentation_clock_pending = bool(getattr(config, "media_presentation_sync", False))
        self._presentation_first_frame_taken = False
        self._presentation_start_index = None
        self._media_cadence_calibrated = False
        self._resume_clock_at = None
        self._paused_clock_pts = None
        self._presentation_lock = threading.Lock()
        self._media_control_lock = threading.RLock()
        self._last_control_revisions = {}
        self._last_publish_at = 0.0
        self._last_image_frame_at = 0.0
        self._position_anchor = self.position
        self._position_anchor_time = None
        self._frame_format = "bgr24"
        self._image = self.mode == "image"
        self._audio_enabled = bool(config.media_audio_enabled)
        self._ready = False
        self._frame_buffer: deque[tuple[float, Any]] = deque()
        self._frame_buffer_condition = threading.Condition()
        try:
            configured_buffer = int(os.environ.get("D2S_MEDIA_FRAME_BUFFER", "8"))
        except (TypeError, ValueError):
            configured_buffer = 8
        self._frame_buffer_capacity = max(2, min(16, configured_buffer))
        self._frame_pump_thread = None
        self._frame_pump_stop = threading.Event()
        self._frame_pump_done = False
        self._frame_pump_error = ""
        self._frame_pump_position = self.position
        self._frame_pump_metrics_lock = threading.Lock()
        self._frame_pump_read_ms = deque(maxlen=120)
        self._frame_pump_frame_times = deque(maxlen=120)
        self._last_source_frame_duration = 0.0
        self._frame_pump_decimation_step = 1
        self._source_selection_schedule = None
        self._last_decimated_emit_at = 0.0
        self._frame_buffer_stale_drops = 0
        self._frame_buffer_drop_log_at = 0.0
        self._decode_lock = threading.RLock()
        try:
            self._open()
            self._browser_mode = False
            self._browser.add_recent(self.path)
        except Exception as exc:
            # File mode is also the entry point for the output-side player.
            # With no saved path yet, opening the browser is the expected
            # initial state, not an error. A stale/missing saved path still
            # surfaces the underlying error beside the browser.
            self.error = (
                ""
                if self._configured_mode.casefold() == "file" and not self.path
                else f"{type(exc).__name__}: {exc}"
            )
            self.playing = False
            if self._configured_mode.casefold() == "file":
                self._browser_mode = True
                self._ready = True
                browser_directory = str(
                    getattr(self.config, "media_browser_directory", "") or ""
                ).strip()
                if not browser_directory or not Path(browser_directory).is_dir():
                    browser_directory = (
                        str(Path(self.path).parent) if self.path else ""
                    )
                self._browser.open(browser_directory or None)
            if self.error:
                print(f"[media] {self.error}", flush=True)
        self._publish_state(force=True)

    @property
    def frame_format(self) -> str:
        return self._frame_format

    @property
    def media_generation(self) -> int:
        """Monotonic identity for seek/reopen invalidation across workers."""
        return int(getattr(self, "_media_generation", 0) or 0)

    @property
    def current_source_frame_identity(self):
        return getattr(self, "_current_source_frame_identity", None)

    def _ensure_frame_pump_metrics(self):
        metrics_lock = getattr(self, "_frame_pump_metrics_lock", None)
        if metrics_lock is None:
            metrics_lock = threading.Lock()
            self._frame_pump_metrics_lock = metrics_lock
        read_values = getattr(self, "_frame_pump_read_ms", None)
        if read_values is None:
            read_values = deque(maxlen=120)
            self._frame_pump_read_ms = read_values
        frame_times = getattr(self, "_frame_pump_frame_times", None)
        if frame_times is None:
            frame_times = deque(maxlen=120)
            self._frame_pump_frame_times = frame_times
        return metrics_lock, read_values, frame_times

    def runtime_metrics(self) -> dict[str, object]:
        """Return bounded producer metrics without touching the frame queue."""
        metrics_lock, read_values, frame_times_value = self._ensure_frame_pump_metrics()
        with metrics_lock:
            read_times = tuple(read_values)
            frame_times = tuple(frame_times_value)
        with self._frame_buffer_condition:
            buffer_depth = len(self._frame_buffer)
            pump_done = bool(self._frame_pump_done)
        ordered = sorted(read_times)

        def percentile(quantile: float) -> float:
            if not ordered:
                return 0.0
            index = min(
                len(ordered) - 1,
                max(0, math.ceil((len(ordered) - 1) * quantile)),
            )
            return float(ordered[index])

        decode_fps = 0.0
        if len(frame_times) >= 2:
            span = frame_times[-1] - frame_times[0]
            if span > 0.0:
                decode_fps = (len(frame_times) - 1) / span
        return {
            "decode_fps": decode_fps,
            "read_ms_p50": percentile(0.50),
            "read_ms_p95": percentile(0.95),
            "read_ms_max": max(ordered) if ordered else 0.0,
            "buffer_depth": buffer_depth,
            "buffer_capacity": self._frame_buffer_capacity,
            "buffer_stale_drops": int(
                getattr(self, "_frame_buffer_stale_drops", 0)
            ),
            "pump_done": pump_done,
            "decode_backend": str(getattr(self, "_decode_backend", "") or ""),
            "fallback_reason": str(getattr(self, "decode_fallback_reason", "") or ""),
        }

    def _open(self) -> None:
        if not self.path:
            raise ValueError("Select a media file before starting the runtime")
        media_path = Path(self.path).expanduser()
        if not media_path.exists():
            raise FileNotFoundError(f"Media source does not exist: {media_path}")
        self.path = str(media_path.resolve())
        if hasattr(self, "_browser"):
            self._browser.prime_media_playlist(self.path)
        if self._image:
            if not media_path.is_file():
                raise ValueError("Select an image file, not a folder or disc")
            self._frame = _load_still_image(self.path)
            self.height, self.width = self._frame.shape[:2]
            self.position = 0.0
            self._position_anchor = 0.0
            self.playing = False
            self._source_frame_index = 0
            self._current_source_frame_identity = (
                self.media_generation,
                0,
                0.0,
            )
            self._ready = True
            return
        if self.mode != "video":
            raise ValueError(f"Unsupported media capture mode: {self.mode}")
        if media_path.is_file() and media_path.suffix.casefold() == ".iso":
            self._mounted_iso_path = str(media_path)
            media_path, self._owns_iso_mount = _mount_iso(media_path)
        if media_path.is_dir():
            entries = _disc_entries(media_path)
            if not entries:
                raise ValueError(f"No playable DVD/Blu-ray title or video found in: {media_path}")
            self._manifest_path = _write_concat_manifest(entries)
            self._input_args = ["-f", "concat", "-safe", "0", "-i", self._manifest_path]
        elif media_path.is_file():
            self._input_args = ["-i", str(media_path)]
            self._nvdec_source_path = str(media_path)
        else:
            raise FileNotFoundError(f"Media source is unavailable: {media_path}")
        self._ffmpeg, ffprobe = _ffmpeg_paths()
        info = _probe_media(ffprobe, self._input_args)
        self._codec_name = info["codec_name"]
        self._amf_decoder = _AMF_DECODERS.get(self._codec_name)
        self._frame_pixel_format = _video_pixel_format(info)
        self._ffmpeg_frame_format = _media_frame_format(info, self._frame_pixel_format)
        self._frame_format = self._ffmpeg_frame_format
        self._video_filter = _video_filter(info, int(self.config.fps))
        self.width = info["width"]
        self.height = info["height"]
        self._decode_width = self.width
        self._decode_height = self.height
        self._amf_output_size = _resolve_amf_output_size(
            info, getattr(self.config, "render_size_config", None)
        )
        self._amf_video_filter = _amf_vpp_filter(
            info, int(self.config.fps), self._amf_output_size
        )
        self.duration = info["duration"]
        self.transfer = info["transfer"]
        self.primaries = info["primaries"]
        self.hdr = self.transfer in _HDR_TRANSFERS
        self._source_fps = float(info["fps"] or 0.0)
        self._playback_fps, self._frame_pump_decimation_step = (
            _resolve_media_playback_cadence(
                self._source_fps,
                int(self.width),
                int(self.height),
                float(self.config.fps),
            )
            if self._source_fps > 0
            else (max(1.0, float(self.config.fps)), 1)
        )
        if self._frame_pump_decimation_step > 1:
            print(
                "[media] deterministic source decimation active: "
                f"source={self._source_fps:.2f}fps "
                f"playback={self._playback_fps:.2f}fps "
                f"step={self._frame_pump_decimation_step}",
                flush=True,
            )
        self._has_audio = info["audio"] and self._audio_enabled
        self.position = min(self.position, max(0.0, self.duration - 0.05))
        self._position_anchor = self.position
        self.playing = True
        self._position_anchor_time = None
        self._source_frame_index = 0
        self._current_source_frame_identity = None
        self._start_video(self.position, realtime=True)
        if self._has_audio:
            self._start_audio(self.position)
        self._ready = True
        self._reset_frame_buffer(self.position)
        self._start_frame_pump()

    def _start_video(self, position: float, *, realtime: bool) -> None:
        self._terminate(self._video_process)
        self._video_process = None
        self._close_nvdec()
        self._close_opencv()
        self._close_native_decoder()
        vendor = getattr(self, "_gpu_vendor", None) or _media_gpu_vendor()
        is_10_bit = getattr(self, "_frame_pixel_format", "") == "p010le"
        if (
            vendor == "nvidia"
            and self._nvdec_source_path
            and not self._nvdec_disabled
        ):
            try:
                if is_10_bit and self.hdr:
                    self._start_nvdec(position, native_hdr=True)
                else:
                    self._start_nvdec(position)
                return
            except Exception as exc:
                self._close_nvdec()
                if not _is_hardware_decode_unavailable(exc):
                    raise
                self._nvdec_disabled = True
                self.decode_fallback_reason = f"{type(exc).__name__}: {exc}"
                print(
                    f"[media] NVDEC unavailable ({self.decode_fallback_reason}); "
                    "falling back to FFmpeg software decoding",
                    flush=True,
                )
        if getattr(self, "_frame_pixel_format", "bgr24") in {"nv12", "p010le"}:
            self._frame_format = getattr(self, "_ffmpeg_frame_format", "bgr24")
        else:
            if self.hdr:
                transfer = "pq" if self.transfer == "smpte2084" else "hlg"
                primaries = self.primaries or "bt2020"
                self._frame_format = f"bgr24_{transfer}_{primaries}"
            elif self.primaries == "bt2020":
                self._frame_format = "bgr24_sdr_bt2020"
            else:
                self._frame_format = "bgr24"
        if vendor == "amd" and self._start_d3d11_video(position):
            return
        if vendor == "amd" and self._amf_decoder and not self._amf_disabled:
            self._decode_backend = "ffmpeg-amf"
            self._start_ffmpeg_video(
                position, realtime=realtime, decoder=self._amf_decoder
            )
            return
        self._decode_backend = "ffmpeg-software"
        self._start_ffmpeg_video(position, realtime=realtime)

    def _start_ffmpeg_video(
        self, position: float, *, realtime: bool, decoder: str | None = None
    ) -> None:
        video_filter = self._video_filter
        amf_output_size = getattr(self, "_amf_output_size", None)
        if decoder and getattr(self, "_amf_video_filter", "") and amf_output_size:
            video_filter = self._amf_video_filter
            self._decode_width, self._decode_height = amf_output_size
        else:
            self._decode_width = int(getattr(self, "width", 0) or 0)
            self._decode_height = int(getattr(self, "height", 0) or 0)
        # Sample source identities once, before host transfer. The frame pump
        # must not decimate these already selected frames a second time.
        source_fps = float(getattr(self, "_source_fps", 0.0) or 0.0)
        playback_fps = float(getattr(self, "_playback_fps", 0.0) or 0.0)
        step = max(1, int(getattr(self, "_frame_pump_decimation_step", 1) or 1))
        selection_schedule = None
        variable_selection = False
        if source_fps > playback_fps + 1e-6:
            selection_schedule = SourceFrameSchedule(source_fps, playback_fps)
            ratio = source_fps / playback_fps
            variable_selection = abs(ratio - round(ratio)) > 1e-6
            step = max(1, int(round(ratio)))
        self._ffmpeg_selection_schedule = selection_schedule
        self._ffmpeg_selection_active = selection_schedule is not None
        self._ffmpeg_decimation_step = step
        self._ffmpeg_first_frame_pending = True
        if selection_schedule is not None:
            filters = [part for part in video_filter.split(",") if not part.startswith("fps=")]
            expression = (
                selection_schedule.ffmpeg_select_expression()
                if variable_selection
                else f"not(mod(n\\,{step}))"
            )
            video_filter = f"select='{expression}'," + ",".join(filters)
        elif step > 1:
            # Worker/unit callers may provide only the legacy decimation step.
            filters = [part for part in video_filter.split(",") if not part.startswith("fps=")]
            video_filter = f"select='not(mod(n\\,{step}))'," + ",".join(filters)
        command = [self._ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
        if decoder:
            command.extend(["-c:v", decoder])
        if realtime:
            command.append("-re")
        if position > 0:
            command.extend(["-ss", f"{position:.3f}"])
        command.extend([
            *self._input_args,
            "-map", "0:v:0",
            "-an",
            "-vf", video_filter,
            "-fps_mode", "passthrough",
            "-pix_fmt", getattr(self, "_frame_pixel_format", "bgr24"),
            "-f", "rawvideo",
            "pipe:1",
        ])
        stderr = tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr,
                bufsize=4 * 1024 * 1024,
            )
        except Exception:
            stderr.close()
            raise
        process._media_stderr = stderr
        self._video_process = process

    def _start_nvdec(self, position: float, *, native_hdr: bool = False) -> None:
        _enable_torch_cuda_dlls()
        import torch

        if getattr(getattr(torch, "version", None), "hip", None):
            raise _HardwareDecodeUnavailable("NVDEC requires NVIDIA CUDA; ROCm is active")
        if not torch.cuda.is_available():
            raise _HardwareDecodeUnavailable("CUDA decoder unavailable: CUDA is not available")
        import PyNvVideoCodec as nvc

        gpu_id = int(torch.cuda.current_device())
        self._nvdec_decode_stream = torch.cuda.Stream(device=gpu_id)
        output_color_type = nvc.OutputColorType.RGBP
        if native_hdr:
            output_color_type = getattr(nvc.OutputColorType, "NATIVE", None)
            if output_color_type is None:
                raise _HardwareDecodeUnavailable(
                    "PyNvVideoCodec NATIVE output is unavailable for HDR decoding"
                )
        decoder_options = {
            "encSource": self._nvdec_source_path,
            "gpuid": gpu_id,
            "cudaStream": self._nvdec_decode_stream.cuda_stream,
            "useDeviceMemory": True,
            "outputColorType": output_color_type,
        }
        threaded_decoder_type = getattr(nvc, "ThreadedDecoder", None)
        if callable(threaded_decoder_type):
            frame_index = 0
            preflight_decoder = nvc.CreateSimpleDecoder(**decoder_options)
            try:
                if position > 0:
                    frame_index = int(
                        preflight_decoder.get_index_from_time_in_seconds(float(position))
                    )
                    preflight_decoder.seek_to_index(frame_index)
                if not preflight_decoder.get_batch_frames(1):
                    raise _HardwareDecodeUnavailable(
                        "NVDEC could not decode a preflight video frame"
                    )
            finally:
                self._dispose_nvdec_decoder(preflight_decoder)
            frame_ratio = (
                self._source_fps / self._playback_fps
                if self._source_fps > 0 and self._playback_fps > 0
                else 1.0
            )
            buffer_size = max(4, min(8, int(frame_ratio) + 1))
            decoder = threaded_decoder_type(
                self._nvdec_source_path,
                buffer_size=buffer_size,
                gpu_id=gpu_id,
                cuda_stream=self._nvdec_decode_stream.cuda_stream,
                use_device_memory=True,
                output_color_type=output_color_type,
                start_frame=frame_index,
            )
            self._nvdec_buffer_size = buffer_size
        else:
            decoder = nvc.CreateSimpleDecoder(**decoder_options)
            self._nvdec_buffer_size = 4
            with self._get_nvdec_lock():
                self._nvdec_decoder = decoder
                self._nvdec_torch = torch
            if position > 0:
                frame_index = decoder.get_index_from_time_in_seconds(float(position))
                decoder.seek_to_index(frame_index)

        with self._get_nvdec_lock():
            self._nvdec_decoder = decoder
            self._nvdec_torch = torch
        first_frame = self._decode_nvdec_frame()
        if first_frame is None:
            raise _HardwareDecodeUnavailable("NVDEC could not decode the first video frame")
        self._frame = first_frame
        self._nvdec_first_frame_pending = True
        self._nvdec_next_frame_at = None
        self._nvdec_skip_fraction = 0.0
        self._decode_backend = "nvdec-native-p010" if native_hdr else "nvdec"
        if self.hdr:
            transfer = "pq" if self.transfer == "smpte2084" else "hlg"
            primaries = self.primaries or "bt2020"
            colorspace = "bt2020nc"
            self._frame_format = (
                f"p010le_{transfer}_{primaries}_{colorspace}_tv_left"
                if native_hdr
                else f"rgbp_{transfer}_{primaries}"
            )
        elif self.primaries == "bt2020":
            self._frame_format = "rgbp_sdr_bt2020"
        else:
            self._frame_format = "rgbp"

    def _get_nvdec_lock(self):
        lock = getattr(self, "_nvdec_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._nvdec_lock = lock
        return lock

    @staticmethod
    def _dispose_nvdec_decoder(decoder) -> None:
        for method_name in ("end", "stop", "close"):
            dispose = getattr(decoder, method_name, None)
            if callable(dispose):
                dispose()
                return

    def _decode_nvdec_frame(self):
        return self._decode_nvdec_frames(1)

    def _decode_nvdec_frames(self, batch_size: int):
        with self._get_nvdec_lock():
            decoder = self._nvdec_decoder
            torch = self._nvdec_torch
            if decoder is None or torch is None:
                return None
            remaining = max(1, int(batch_size))
            requested_total = remaining
            buffer_size = max(1, int(getattr(self, "_nvdec_buffer_size", remaining)))
            frame = None
            while remaining > 0:
                requested = min(remaining, buffer_size)
                decoded = decoder.get_batch_frames(requested)
                if not decoded:
                    break
                remaining -= len(decoded)
                if remaining > 0 and len(decoded) >= requested:
                    del decoded  # Release discarded decoder surfaces before the next ring fetch.
                    continue
                copy_stream = getattr(self, "_nvdec_copy_stream", None)
                if copy_stream is None:
                    copy_stream = torch.cuda.Stream(device=torch.cuda.current_device())
                    self._nvdec_copy_stream = copy_stream
                decode_stream = getattr(self, "_nvdec_decode_stream", None)
                if decode_stream is not None:
                    copy_stream.wait_stream(decode_stream)
                with torch.cuda.stream(copy_stream):
                    decoded_frame = torch.from_dlpack(decoded[-1])
                    if decoded_frame.ndim == 4 and decoded_frame.shape[0] == 1:
                        decoded_frame = decoded_frame[0]
                    frame_dtype = getattr(decoded_frame, "dtype", None)
                    native_p010 = (
                        decoded_frame.ndim == 2
                        and frame_dtype == torch.uint16
                    )
                    if not native_p010 and (
                        decoded_frame.ndim != 3 or decoded_frame.shape[0] != 3
                    ):
                        raise _HardwareDecodeUnavailable(
                            "NVDEC returned an unsupported frame shape: "
                            f"{tuple(decoded_frame.shape)} dtype={frame_dtype}"
                        )
                    if decoded_frame.device.type != "cuda":
                        raise _HardwareDecodeUnavailable(
                            f"NVDEC did not return CUDA device memory: {decoded_frame.device}"
                        )
                    if native_p010 and tuple(decoded_frame.stride()) != (
                        decoded_frame.shape[1],
                        1,
                    ):
                        # PyNvVideoCodec's NATIVE DLPack view reports a byte-like
                        # column stride for the packed P010 surface. The storage
                        # itself is row-major; normalize the view before the
                        # Triton YUV kernel consumes it.
                        decoded_frame = decoded_frame.as_strided(
                            tuple(decoded_frame.shape),
                            (decoded_frame.shape[1], 1),
                        )
                    # Own the frame before the next decoder read can recycle its surface.
                    frame = decoded_frame.clone()
                # Wait only for the owned copy, not unrelated default-stream inference.
                copy_stream.synchronize()
                if len(decoded) < requested:
                    break
            self._nvdec_last_frame_advance = requested_total - remaining
            return frame

    def _close_nvdec(self) -> None:
        with self._get_nvdec_lock():
            decoder = self._nvdec_decoder
            self._nvdec_decoder = None
            self._nvdec_torch = None
            self._nvdec_first_frame_pending = False
            self._decoder_resampled = True
            self._decoder_frame_advance = 1
            self._nvdec_next_frame_at = None
            if decoder is not None:
                self._dispose_nvdec_decoder(decoder)
            stream = getattr(self, "_nvdec_decode_stream", None)
            if stream is not None:
                stream.synchronize()
                self._nvdec_decode_stream = None

    def _start_audio(self, position: float) -> None:
        try:
            import sounddevice as sd

            sync_enabled = bool(getattr(self.config, "media_presentation_sync", False))
            command = [self._ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
            if not sync_enabled:
                command.append("-re")
            if position > 0:
                command.extend(["-ss", f"{position:.9f}"])
            command.extend([
                *self._input_args, "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "48000",
                "-af", "aresample=async=1:first_pts=0",
                "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
            ])
            self._audio_process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._audio_stop = threading.Event()
            sync = None
            stream_type = sd.RawOutputStream
            stream_kwargs = {"samplerate": 48000, "channels": 2, "dtype": "float32", "blocksize": 0}
            if sync_enabled:
                from .media_audio import MediaAudioSync

                sync = MediaAudioSync(generation=self.media_generation, start_pts=float(position))
                self._audio_sync = sync
                stream_type = sd.OutputStream
                def audio_callback(outdata, frames, time_info, status):
                    if status.output_underflow:
                        sync.note_device_underflow()
                    callback_at = time.perf_counter()
                    # Pa_GetStreamTime is calibrated once to perf_counter.
                    # Mapping from callback entry/currentTime on every block
                    # would add Python/GIL scheduling delay to the DAC PTS.
                    dac_at = float(time_info.outputBufferDacTime) + audio_clock_offset
                    sync.render(outdata, dac_at=dac_at, callback_at=callback_at, volume=self.volume, muted=self.muted)
                stream_kwargs.update(blocksize=480, latency="low", callback=audio_callback)
            else:
                self._audio_sync = None
            preferred_device = _preferred_audio_output_device(sd)
            if preferred_device is not None:
                stream_kwargs["device"] = preferred_device
            try:
                self._audio_stream = stream_type(**stream_kwargs)
            except Exception:
                if preferred_device is None:
                    raise
                stream_kwargs.pop("device", None)
                self._audio_stream = stream_type(**stream_kwargs)
            audio_clock_offset = (
                time.perf_counter() - float(self._audio_stream.time) if sync_enabled else 0.0
            )
            self._audio_stream.start()
            self._audio_stop.clear()
            self._audio_position_anchor = max(0.0, float(position))
            self._audio_started_at = None if sync_enabled else time.perf_counter()
            self._audio_thread = threading.Thread(
                target=self._audio_loop, name="MediaAudioOutput", daemon=True
            )
            self._audio_thread.start()
            self.audio_error = ""
        except Exception as exc:
            self.audio_error = f"{type(exc).__name__}: {exc}"
            self._close_audio()
            print(f"[media] audio output unavailable: {self.audio_error}", flush=True)

    def _audio_loop(self) -> None:
        process, stream = self._audio_process, self._audio_stream
        stop = self._audio_stop
        sync = getattr(self, "_audio_sync", None)
        try:
            while not stop.is_set() and process is not None:
                # A pipe read may split a float or channel pair. Assemble exact
                # 10 ms PCM blocks; never invent sample PTS from read timing.
                data = bytearray()
                block_bytes = 480 * 2 * 4 if sync is not None else 16 * 1024
                while len(data) < block_bytes and not stop.is_set():
                    chunk = process.stdout.read(block_bytes - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                complete_bytes = len(data) - len(data) % 8
                if not complete_bytes or stop.is_set():
                    if not stop.is_set() and sync is not None:
                        status = process.poll() if callable(getattr(process, "poll", None)) else None
                        wait = getattr(process, "wait", None)
                        if status is None and callable(wait):
                            status = wait(timeout=0.5)
                        if status not in (None, 0):
                            raise RuntimeError(f"FFmpeg audio decoder exited with status {status}")
                        note_eof = getattr(sync, "note_source_eof", None)
                        if callable(note_eof):
                            note_eof()
                    break
                samples = np.frombuffer(data, dtype=np.float32, count=complete_bytes // 4).copy()
                if sync is not None:
                    if not sync.enqueue(samples.reshape(-1, 2), stop):
                        break
                    continue
                gain = 0.0 if self.muted else max(0.0, min(1.0, self.volume))
                if gain != 1.0:
                    samples *= gain
                if stream is not None:
                    stream.write(samples)
        except Exception as exc:
            if stop.is_set():
                return
            self.audio_error = f"{type(exc).__name__}: {exc}"
            self._audio_stop.set()
            process, self._audio_process = self._audio_process, None
            self._terminate(process)
            stream, self._audio_stream = self._audio_stream, None
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            print(f"[media] audio output stopped: {self.audio_error}", flush=True)

    @staticmethod
    def _terminate(process) -> None:
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)
        for stream in (process.stdout, getattr(process, "_media_stderr", None)):
            if stream is not None and not stream.closed:
                stream.close()

    def _close_audio(self) -> None:
        self._audio_stop.set()
        sync, self._audio_sync = getattr(self, "_audio_sync", None), None
        if sync is not None:
            sync.close()
        self._audio_started_at = None
        process, self._audio_process = self._audio_process, None
        self._terminate(process)
        stream, self._audio_stream = self._audio_stream, None
        if stream is not None:
            try:
                abort = getattr(stream, "abort", stream.stop)
                abort()
                stream.close()
            except Exception:
                pass
        thread, self._audio_thread = self._audio_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

    def observe_video_presentation(self, generation: int, index: int, pts: float, at: float) -> None:
        """Called after presentation; do not run decoder/audio setup on the XR thread."""
        if generation != self.media_generation or not math.isfinite(pts) or not math.isfinite(at):
            return
        with self._presentation_lock:
            if self.playing and getattr(self, "_presentation_clock_pending", False):
                start_index = getattr(self, "_presentation_start_index", None)
                if start_index is None or index < start_index:
                    return  # A pre-pause queued output cannot start the new playback epoch.
            previous = self._presented_media_identity
            if previous is not None and previous[0] == generation and index <= previous[1]:
                return
            self._presented_media_identity = (int(generation), int(index), float(pts), float(at))
            if self.playing and getattr(self, "_presentation_clock_pending", False):
                # Loading, shader compilation and cadence preflight are preroll,
                # not part of the source playback duration.
                resume_at = getattr(self, "_resume_clock_at", None)
                self._position_anchor = float(pts) if resume_at is None else float(self.position)
                self._position_anchor_time = float(at) if resume_at is None else float(resume_at)
                self._resume_clock_at = None
                self._cadence_observe_after = float(at) + 2.0
                self._presentation_clock_pending = False
                with self._frame_buffer_condition:
                    self._frame_buffer_condition.notify_all()
            sync = self._audio_sync
            if sync is not None and self.playing:
                sync.observe_video(generation, index, pts, at, self._frame_duration())
            if not self.playing:
                self.position = float(pts)
                self._position_anchor = self.position

    def calibrate_media_cadence(self, service_ms: float) -> None:
        """Select an initial measured divisor before any source clock advances."""
        if (not getattr(self, "_presentation_clock_pending", False) or self._media_cadence_calibrated
                or not getattr(self, "_ready", True) or self._source_fps <= 0.0):
            return
        if not math.isfinite(service_ms) or service_ms <= 0.0:
            return
        fps, step = _resolve_media_playback_cadence(
            self._source_fps, self.width, self.height, float(self.config.fps),
            target_fps=1000.0 / service_ms,
        )
        with self._frame_buffer_condition:
            self._playback_fps = fps
            self._frame_pump_decimation_step = step
            self._last_source_frame_duration = 1.0 / fps
            self._media_prepare_lead_s = service_ms * 0.001 + 1.0 / max(1.0, float(self.config.fps))
            self._media_cadence_calibrated = True
        print(f"[media] cadence preroll: service={service_ms:.2f}ms target={fps:.2f}fps step={step}", flush=True)

    def observe_media_timing(self, metadata: dict, at: float) -> None:
        if (not self.playing or getattr(self, "_browser_mode", False)
                or getattr(self, "_presentation_clock_pending", False)):
            return
        if str(os.environ.get("D2S_MEDIA_CADENCE_ADAPTIVE", "1")).strip().lower() in {
            "0", "false", "no", "off"
        }:
            return
        generation = metadata.get("media_generation")
        identity = (generation, metadata.get("source_frame_index"))
        if generation == self.media_generation and identity != getattr(self, "_media_timing_identity", None):
            previous = getattr(self, "_media_timing_identity", None)
            self._media_timing_identity = identity
            captured_at = metadata.get("capture_poll_returned_at")
            if previous is not None and captured_at is not None:
                latency = float(at) - float(captured_at)
                if math.isfinite(latency) and 0.0 <= latency <= 0.25:
                    samples = getattr(self, "_media_prepare_latency_samples", None)
                    if samples is None:
                        samples = self._media_prepare_latency_samples = deque(maxlen=17)
                    samples.append(latency)
                    self._media_prepare_lead_s = sorted(samples)[len(samples) // 2]
        if (float(at) < float(getattr(self, "_cadence_observe_after", 0.0))
                or abs(float(metadata.get("media_playback_fps") or self._playback_fps) - self._playback_fps) > 0.01):
            return
        if generation != self.media_generation or metadata.get("source_pts") is None:
            return
        from .media_cadence import MediaCadenceController

        controller = getattr(self, "_media_cadence_controller", None)
        if controller is None or getattr(self, "_media_cadence_generation", None) != generation:
            ceiling, _step = _resolve_media_playback_cadence(
                self._source_fps, self.width, self.height, float(self.config.fps)
            )
            controller = MediaCadenceController(self._source_fps, ceiling)
            self._media_cadence_controller = controller
            self._media_cadence_generation = generation
            self._adaptive_cadence_target = None
        metrics = self.runtime_metrics()
        target = controller.observe(
            pts=float(metadata["source_pts"]), at=float(at),
            playback_fps=float(metadata.get("media_playback_fps") or self._playback_fps),
            service_ms=max(float(metadata.get("runtime_completion_ms") or 0.0),
                           float(metadata.get("vk_total_ms") or 0.0)),
            decode_ms=float(metrics.get("read_ms_p95") or 0.0),
            buffer_depth=int(metrics.get("buffer_depth") or 0),
            display_fps=float(metadata.get("media_display_fps") or 0.0),
            display_vsync=bool(metadata.get("media_display_vsync", True)),
            lateness_ms=(max(0.0, (float(at) - float(metadata["media_presentation_deadline"])) * 1000.0)
                         if metadata.get("media_presentation_deadline") is not None else 0.0),
        )
        if target is not None:
            self._adaptive_cadence_target = target
            print(f"[media] adaptive cadence: target={target:.2f}fps reason={controller.reason}", flush=True)

    def media_presentation_clock(self) -> dict:
        """Return the live presentation epoch, including pause/resume ownership."""
        return {
            "generation": self.media_generation,
            "playing": bool(self.playing),
            "anchor_at": self._position_anchor_time if self._position_anchor_time is not None
            else getattr(self, "_resume_clock_at", None),
            "anchor_pts": self._position_anchor,
            "paused_at": getattr(self, "_paused_clock_at", None),
            "resumed_at": getattr(self, "_last_resume_clock_at", None),
        }

    def audio_sync_metrics(self) -> dict:
        sync = getattr(self, "_audio_sync", None)
        return sync.snapshot(time.perf_counter()) if sync is not None else {}

    def _current_position(self) -> float:
        if self.playing and self._position_anchor_time is not None:
            value = (
                self._position_anchor
                + time.perf_counter()
                - self._position_anchor_time
            )
            return min(value, self.duration) if self.duration > 0 else value
        return self.position

    def _frame_duration(self) -> float:
        return 1.0 / max(1.0, float(getattr(self, "_playback_fps", 1.0) or 1.0))


    def _source_frame_duration(self) -> float:
        """Return the period of frames actually emitted by the decoder.

        FFmpeg's fps filter already reduces 120 fps input to the configured
        playback cadence, while the native D3D11 ring emits every source frame.
        The pump records that real cadence so stale-frame rejection uses the
        correct interval in both paths.
        """
        try:
            duration = float(getattr(self, "_last_source_frame_duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if math.isfinite(duration) and duration > 0.0:
            return duration
        return self._frame_duration()

    def _next_source_frame_advance(self) -> int:
        """Return the next ordered source-frame step for the live cadence."""
        source_fps = float(getattr(self, "_source_fps", 0.0) or 0.0)
        playback_fps = float(getattr(self, "_playback_fps", 1.0) or 1.0)
        schedule = getattr(self, "_source_selection_schedule", None)
        if (
            schedule is None
            or abs(schedule.rates[0] - source_fps) > 1e-4
            or abs(schedule.rates[1] - playback_fps) > 1e-4
        ):
            schedule = SourceFrameSchedule(source_fps, playback_fps)
            self._source_selection_schedule = schedule
        return schedule.next_advance()

    def _close_native_decoder(self) -> None:
        importer = getattr(self, "_native_frame_importer", None)
        decoder = getattr(self, "_native_decoder", None)
        self._native_frame_importer = None
        self._native_decoder = None
        if importer is not None:
            try:
                importer.close()
            except Exception:
                pass
        if decoder is not None:
            try:
                decoder.close()
            except Exception:
                pass
        self._native_frame_pts = None
        self._native_frame_duration = 0.0

    def _start_d3d11_video(self, position: float) -> bool:
        if os.name != "nt" or not getattr(self, "_nvdec_source_path", None):
            return False
        source_path = Path(self._nvdec_source_path)
        if not source_path.is_file():
            return False
        if str(os.environ.get("D2S_MEDIA_D3D11_DECODER", "1")).strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return False
        try:
            from capture.media_d3d11 import (
                D3D11VideoFrameImporter,
                FFmpegD3D11Decoder,
                d3d11_decoder_available,
            )

            if not d3d11_decoder_available():
                return False
            decoder = FFmpegD3D11Decoder(
                self._nvdec_source_path,
                start_seconds=position,
                ring_size=8,
            )
            try:
                importer = D3D11VideoFrameImporter()
            except Exception:
                decoder.close()
                raise
            self._native_decoder = decoder
            self._native_frame_importer = importer
            self._decode_backend = "ffmpeg-d3d11va-gpu"
            self._frame_format = "bgr24_sdr_bt709"
            self.decode_fallback_reason = ""
            print(
                "[media] GPU video path active: "
                f"backend={self._decode_backend} source={self._source_fps:.2f}fps "
                "transfer=d3d11_video_processor_sdr",
                flush=True,
            )
            return True
        except Exception as exc:
            self.decode_fallback_reason = f"{type(exc).__name__}: {exc}"
            print(
                f"[media] D3D11VA GPU decode unavailable "
                f"({self.decode_fallback_reason}); using FFmpeg fallback",
                flush=True,
            )
            return False

    def _reset_frame_buffer(self, position: float) -> None:
        self._media_cadence_controller = None
        self._adaptive_cadence_target = None
        self._source_selection_schedule = None
        self._presentation_clock_pending = bool(getattr(getattr(self, "config", None), "media_presentation_sync", False))
        self._presentation_first_frame_taken = False
        self._presentation_start_index = None
        self._media_cadence_calibrated = False
        self._resume_clock_at = None
        self._paused_clock_pts = None
        self._media_prepare_lead_s = 0.0
        self._media_prepare_latency_samples = deque(maxlen=17)
        self._media_timing_identity = None
        with self._frame_buffer_condition:
            self._frame_buffer.clear()
            self._frame_pump_done = False
            self._frame_pump_error = ""
            self._frame_pump_position = max(0.0, float(position))
            self._native_frame_pts = None
            self._native_frame_duration = 0.0
            self._last_source_frame_duration = 0.0
            self._current_source_frame_identity = None
            self._last_decimated_emit_at = 0.0
            if hasattr(self, "_frame_pump_metrics_lock"):
                with self._frame_pump_metrics_lock:
                    self._frame_pump_read_ms.clear()
                    self._frame_pump_frame_times.clear()
            self._frame_buffer_condition.notify_all()

    def _start_frame_pump(self) -> None:
        """Keep decoding ahead of the inference consumer on a bounded queue."""
        if self._image or not self._ready or not self.playing:
            return
        thread = self._frame_pump_thread
        if thread is not None and thread.is_alive():
            return
        with self._frame_buffer_condition:
            if self._frame_pump_done:
                return
            self._frame_pump_stop.clear()
            thread = threading.Thread(
                target=self._frame_pump_loop,
                name="MediaFramePump",
                daemon=True,
            )
            self._frame_pump_thread = thread
        thread.start()
        print(
            "[media] realtime frame buffer active: "
            f"capacity={self._frame_buffer_capacity} "
            f"playback_fps={self._playback_fps:.2f}",
            flush=True,
        )

    def _stop_frame_pump(self) -> None:
        self._frame_pump_stop.set()
        with self._frame_buffer_condition:
            self._frame_buffer_condition.notify_all()
        thread = self._frame_pump_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                print(
                    "[media] frame pump did not stop before decoder restart",
                    flush=True,
                )
        self._frame_pump_thread = None

    def _apply_media_playback_cadence_target(self) -> None:
        target = getattr(self, "_adaptive_cadence_target", None) or get_media_playback_cadence_target()
        if target is None or self._source_fps <= 0.0:
            return
        # The adaptive controller has already selected a display-aligned rate.
        # Keep that rate exactly; rounding it through an integer source divisor
        # would turn 36 FPS into 30 FPS and reintroduce the 2/3-tick pattern.
        playback_fps = min(float(self._source_fps), max(1.0, float(target)))
        step = max(1, int(round(float(self._source_fps) / playback_fps)))
        if (
            abs(playback_fps - float(self._playback_fps)) < 0.01
            and int(step) == int(self._frame_pump_decimation_step)
        ):
            return
        with self._frame_buffer_condition:
            if getattr(self, "_adaptive_cadence_target", None) is not None:
                # Apply a new divisor prospectively. Drain every already
                # selected source frame instead of jumping across a full queue.
                while self._frame_buffer and not self._frame_pump_stop.is_set():
                    self._frame_buffer_condition.wait(timeout=0.05)
                if self._frame_pump_stop.is_set():
                    return
            previous_fps = float(self._playback_fps)
            previous_step = int(self._frame_pump_decimation_step)
            self._playback_fps = float(playback_fps)
            self._frame_pump_decimation_step = int(step)
            self._source_selection_schedule = None
            self._nvdec_skip_fraction = 0.0
            self._opencv_skip_fraction = 0.0
            self._last_source_frame_duration = 1.0 / max(1.0, float(playback_fps))
            self._last_decimated_emit_at = 0.0
            # FFmpeg selects before the raw pipe. Rebuild that selector at the
            # oldest unconsumed PTS, rather than silently dropping its queue.
            restart_pipe = (
                getattr(self, "_native_decoder", None) is None
                and str(getattr(self, "_decode_backend", "")).startswith("ffmpeg")
            )
            if restart_pipe and self._frame_buffer:
                self._frame_pump_position = float(self._frame_buffer[0][0])
                self._source_frame_index = int(self._frame_buffer[0][2])
            self._frame_buffer.clear()
            controller = getattr(self, "_media_cadence_controller", None)
            if controller is not None:
                controller.reset_window()
            self._frame_buffer_condition.notify_all()
        if restart_pipe:
            decoder = self._amf_decoder if self._decode_backend == "ffmpeg-amf" else None
            self._start_ffmpeg_video(self._frame_pump_position, realtime=True, decoder=decoder)
        print(
            "[media] playback cadence retargeted: "
            f"target={target:.2f}fps previous={previous_fps:.2f}fps/step{previous_step} "
            f"playback={playback_fps:.2f}fps/step{step}",
            flush=True,
        )

    def _read_buffered_video_frame(self):
        importer = getattr(self, "_native_frame_importer", None)
        if importer is None:
            return self._read_video_frame()
        import torch

        # Future decode-ahead fences must not interleave with inference kernels.
        # Publish an owned, completed GPU copy, without reading pixels to CPU.
        stream = importer._stream
        with torch.cuda.stream(stream):
            frame = self._read_video_frame()
            if frame is not None and getattr(self, "_native_decoder", None) is not None:
                frame = frame.clone()
        stream.synchronize()
        return frame

    def _frame_pump_loop(self) -> None:
        try:
            while not self._frame_pump_stop.is_set():
                with self._frame_buffer_condition:
                    while (
                        (len(self._frame_buffer) >= self._frame_buffer_capacity
                         or (getattr(self, "_presentation_clock_pending", False) and self._source_frame_index > 0))
                        and not self._frame_pump_stop.is_set()
                    ):
                        self._frame_buffer_condition.wait(timeout=0.05)
                if self._frame_pump_stop.is_set():
                    break
                self._apply_media_playback_cadence_target()
                with self._decode_lock:
                    read_started_at = time.perf_counter()
                    self._decoder_frame_advance = 1
                    self._decoder_resampled = False
                    self._decoder_terminal_partial = False
                    frame = self._read_buffered_video_frame()
                    read_ms = max(0.0, (time.perf_counter() - read_started_at) * 1000.0)
                now = time.perf_counter()
                advance = max(1, int(getattr(self, "_decoder_frame_advance", 1)))
                source_index = int(self._source_frame_index) + advance - 1
                decoder_resampled = bool(getattr(self, "_decoder_resampled", False))
                decimation_step = max(
                    1, int(getattr(self, "_frame_pump_decimation_step", 1) or 1)
                )
                emit_frame = frame is not None and (decoder_resampled or source_index % decimation_step == 0)
                metrics_lock, read_values, frame_times = self._ensure_frame_pump_metrics()
                with metrics_lock:
                    read_values.append(read_ms)
                    if emit_frame:
                        frame_times.append(now)
                if frame is None:
                    with self._frame_buffer_condition:
                        self._frame_pump_done = True
                        self._frame_buffer_condition.notify_all()
                    break
                with self._frame_buffer_condition:
                    # A raw decoder tick is a SOURCE period, not the slower
                    # selected playback period. Batched readers report how many
                    # original frames they consumed to produce this image.
                    frame_duration = (
                        1.0 / self._source_fps
                        if getattr(self, "_source_fps", 0.0) > 0.0 and (decoder_resampled or decimation_step > 1)
                        else self._frame_duration()
                    )
                    source_pts = self._frame_pump_position + (advance - 1) * frame_duration
                    native_pts = getattr(self, "_native_frame_pts", None)
                    if native_pts is not None:
                        source_pts = native_pts
                        frame_duration = getattr(
                            self, "_native_frame_duration", 0.0
                        ) or frame_duration
                        self._native_frame_pts = None
                        self._native_frame_duration = 0.0
                    self._last_source_frame_duration = max(
                        0.0, float(frame_duration) * (advance if decoder_resampled else decimation_step)
                    )
                    duration = float(getattr(self, "duration", 0.0) or 0.0)
                    if duration > 0.0 and source_pts >= duration:
                        # A partial/VFR decoder tail must not leave an
                        # unpresentable post-duration item blocking ordered EOF.
                        self._frame_pump_done = True
                        self._frame_buffer_condition.notify_all()
                        break
                    if emit_frame:
                        self._frame_buffer.append(
                            (
                                source_pts,
                                frame,
                                source_index,
                                {"media_playback_fps": float(self._playback_fps),
                                 "media_terminal_partial": bool(getattr(self, "_decoder_terminal_partial", False))},
                            )
                        )
                    self._source_frame_index = source_index + 1
                    self._frame_pump_position = source_pts + frame_duration
                    self._frame_buffer_condition.notify_all()
        except Exception as exc:
            with self._frame_buffer_condition:
                self._frame_pump_error = f"{type(exc).__name__}: {exc}"
                self._frame_pump_done = True
                self._frame_buffer_condition.notify_all()

    def _take_buffered_frame(
        self,
        target_position: float,
        *,
        allow_first: bool = False,
    ):
        playback_period = self._frame_duration()
        source_period = self._source_frame_duration()
        tolerance = 0.5 * min(playback_period, source_period)
        # When the decoder already emits at the playback cadence, small consumer
        # jitter is recoverable: return the oldest due source frame and let the
        # next grab catch up. Selecting the newest due frame here drops source
        # frames every time one inference tick slips, which reads as a jump.
        preserve_source_order = source_period >= playback_period * 0.95
        max_exact_lateness = 3.0 * max(playback_period, source_period)
        wait_budget = min(0.02, max(0.002, source_period * 1.5))
        deadline = time.perf_counter() + wait_budget
        with self._frame_buffer_condition:
            while True:
                if self._frame_buffer and allow_first:
                    item = self._frame_buffer.popleft()
                    if int(getattr(self, "_frame_pump_decimation_step", 1) or 1) > 1:
                        self._last_decimated_emit_at = time.perf_counter()
                    self._frame_buffer_condition.notify_all()
                    return item

                if bool(getattr(getattr(self, "config", None), "media_presentation_sync", False)):
                    # Absolute source time already owns selection. A second
                    # minimum-emit phase retains any temporary GPU stall as
                    # permanent slow playback, even after the rate downshifts.
                    if self._frame_buffer and float(self._frame_buffer[0][0]) <= target_position + tolerance:
                        item = self._frame_buffer.popleft()
                        self._frame_buffer_condition.notify_all()
                        return item
                    if self._frame_pump_done and not self._frame_buffer:
                        return None
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        return None
                    self._frame_buffer_condition.wait(timeout=min(0.005, remaining))
                    continue

                if int(getattr(self, "_frame_pump_decimation_step", 1) or 1) > 1:
                    playback_period = self._frame_duration()
                    now = time.perf_counter()
                    pace_ready = (
                        self._last_decimated_emit_at <= 0.0
                        or now - self._last_decimated_emit_at
                        >= playback_period - min(0.001, playback_period * 0.05)
                    )
                    if self._frame_buffer and pace_ready:
                        candidate_pts = float(self._frame_buffer[0][0])
                        if candidate_pts <= target_position + tolerance:
                            item = self._frame_buffer.popleft()
                            previous_deadline = self._last_decimated_emit_at
                            self._last_decimated_emit_at = (
                                previous_deadline + playback_period
                                if previous_deadline > 0.0 and now - previous_deadline < 3.0 * playback_period
                                else now
                            )
                            self._frame_buffer_condition.notify_all()
                            return item
                    if self._frame_pump_done and not self._frame_buffer:
                        return None
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        return None
                    self._frame_buffer_condition.wait(
                        timeout=min(0.005, remaining)
                    )
                    continue

                if preserve_source_order:
                    dropped_stale = 0
                    while (
                        self._frame_buffer
                        and target_position - float(self._frame_buffer[0][0])
                        > max_exact_lateness
                    ):
                        self._frame_buffer.popleft()
                        dropped_stale += 1
                    if dropped_stale:
                        self._frame_buffer_stale_drops += dropped_stale
                        now = time.perf_counter()
                        if now - self._frame_buffer_drop_log_at >= 2.0:
                            self._frame_buffer_drop_log_at = now
                            print(
                                "[media] bounded catch-up dropped "
                                f"{dropped_stale} stale source frame(s)",
                                flush=True,
                            )
                    if self._frame_buffer:
                        candidate_pts = float(self._frame_buffer[0][0])
                        if candidate_pts <= target_position + tolerance:
                            item = self._frame_buffer.popleft()
                            self._frame_buffer_condition.notify_all()
                            return item
                    if self._frame_pump_done:
                        return None
                else:
                    # Keep exactly the newest source frame whose PTS has reached
                    # the wall-clock media position when the decoder is
                    # intentionally resampling a higher source cadence.
                    while (
                        len(self._frame_buffer) > 1
                        and self._frame_buffer[1][0] <= target_position
                    ):
                        self._frame_buffer.popleft()

                    if self._frame_buffer:
                        candidate = self._frame_buffer[0]
                        candidate_pts = float(candidate[0])
                        latest_is_proven = (
                            len(self._frame_buffer) > 1
                            and float(self._frame_buffer[1][0]) > target_position
                        )
                        close_enough = (
                            candidate_pts <= target_position
                            and target_position - candidate_pts <= tolerance
                        )
                        if candidate_pts <= target_position and (
                            latest_is_proven or close_enough
                        ):
                            item = self._frame_buffer.popleft()
                            self._frame_buffer_condition.notify_all()
                            return item
                        if self._frame_pump_done:
                            if candidate_pts <= target_position:
                                item = self._frame_buffer.popleft()
                                self._frame_buffer_condition.notify_all()
                                return item
                            return None
                    elif self._frame_pump_done:
                        return None

                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    # Hold the last valid presented frame and retry on the
                    # next consumer tick rather than advancing to stale media.
                    return None
                self._frame_buffer_condition.wait(timeout=min(0.005, remaining))

    def _has_open_video_decoder(self) -> bool:
        if getattr(self, "_native_decoder", None) is not None:
            return True
        if str(getattr(self, "_decode_backend", "")).startswith("nvdec"):
            return getattr(self, "_nvdec_decoder", None) is not None
        if str(getattr(self, "_decode_backend", "")) == "opencv-software":
            return getattr(self, "_opencv_capture", None) is not None
        process = getattr(self, "_video_process", None)
        return process is not None and process.poll() is None

    def _pause_playback(self) -> None:
        # Freeze source time within the displayed frame interval. The visible
        # image keeps its exact PTS; resuming must not skip the remaining part
        # of that interval by rebasing to the next selected image's PTS.
        self._paused_clock_pts = float(self._current_position())
        self._paused_clock_at = (
            float(self._position_anchor_time) + self._paused_clock_pts - float(self._position_anchor)
            if self._position_anchor_time is not None else time.perf_counter()
        )
        controller = getattr(self, "_media_cadence_controller", None)
        if controller is not None:
            controller.reset_window()
        presented = getattr(self, "_presented_media_identity", None)
        self.position = (
            presented[2]
            if presented is not None and presented[0] == self.media_generation
            else self._current_position()
        )
        self._position_anchor = self.position
        self._position_anchor_time = None
        self.playing = False
        if hasattr(self, "_audio_stop"):
            self._close_audio()
        # Keep the bounded frame queue and decoder alive. The next Play command
        # can resume from the buffered frame instead of waiting for a cold seek.
        if hasattr(self, "_frame_buffer_condition"):
            self._stop_frame_pump()

    def _resume_playback(self, position: float) -> None:
        self._browser_mode = False
        self.playing = True
        synced = bool(getattr(self.config, "media_presentation_sync", False))
        paused_pts = getattr(self, "_paused_clock_pts", None)
        if synced and paused_pts is not None:
            position = float(paused_pts)
        elif not synced:
            with self._frame_buffer_condition:
                if self._frame_buffer:
                    position = max(position, self._frame_buffer[0][0])
        self.position = max(0.0, float(position))
        self._position_anchor = self.position
        self._position_anchor_time = None
        self._resume_clock_at = time.perf_counter() if synced else None
        self._last_resume_clock_at = self._resume_clock_at
        self._presentation_clock_pending = synced
        self._presentation_first_frame_taken = False
        presented = getattr(self, "_presented_media_identity", None)
        self._presentation_start_index = (
            int(presented[1]) + 1 if presented is not None and presented[0] == self.media_generation else None
        )
        if getattr(self, "_has_audio", False):
            self._start_audio(self.position)
        self._start_frame_pump()

    def _restart(self, position: float, *, playing: bool) -> None:
        self._media_generation = self.media_generation + 1
        has_frame_buffer = hasattr(self, "_frame_buffer_condition")
        if has_frame_buffer:
            self._stop_frame_pump()
            self._reset_frame_buffer(position)
        self._source_frame_index = 0
        self._current_source_frame_identity = None
        self._terminate(self._video_process)
        self._video_process = None
        self._close_audio()
        self._eof = False
        self.position = max(0.0, min(float(position), self.duration or float(position)))
        self._position_anchor = self.position
        self.playing = bool(playing)
        self._position_anchor_time = None
        self._frame = None
        self._displayed_frame = None
        if self.playing:
            self._start_video(self.position, realtime=True)
            if getattr(self, "_has_audio", False):
                self._start_audio(self.position)
            if has_frame_buffer:
                self._start_frame_pump()
        else:
            self._start_video(self.position, realtime=False)
            if self._decode_backend.startswith("nvdec"):
                self._close_nvdec()
            else:
                self._read_video_frame()
                self._terminate(self._video_process)
                self._video_process = None
                self._close_opencv()

    def _reset_media_for_open(self, path: str, mode: str, position: float) -> None:
        self._media_generation = self.media_generation + 1
        self._source_frame_index = 0
        self._current_source_frame_identity = None
        self.path = path
        self.mode = mode
        self._image = mode == "image"
        self.position = 0.0 if self._image else position
        self._position_anchor = self.position
        self._position_anchor_time = None
        self.playing = False
        self.width = 0
        self.height = 0
        self.duration = 0.0
        self.hdr = False
        self.transfer = ""
        self.primaries = ""
        self.error = ""
        self.audio_error = ""
        self._frame = None
        self._displayed_frame = None
        if hasattr(self, "_frame_buffer_condition"):
            self._reset_frame_buffer(self.position)
        self._ffmpeg = None
        self._video_filter = ""
        self._amf_video_filter = ""
        self._amf_output_size = None
        self._decode_width = 0
        self._decode_height = 0
        self._frame_pixel_format = "bgr24"
        self._ffmpeg_frame_format = "bgr24"
        self._input_args = []
        self._nvdec_source_path = None
        self._nvdec_disabled = False
        self._has_audio = False
        self._codec_name = ""
        self._amf_decoder = None
        self._amf_disabled = False
        self._nvdec_skip_fraction = 0.0
        self._decode_backend = "ffmpeg-software"
        self.decode_fallback_reason = ""
        self._source_fps = 0.0
        self._playback_fps = max(1.0, float(self.config.fps))
        self._manifest_path = None
        self._mounted_iso_path = None
        self._owns_iso_mount = False
        self._opencv_skip_fraction = 0.0
        self._opencv_realtime = False
        self._audio_position_anchor = self.position
        self._last_image_frame_at = 0.0
        self._frame_format = "bgr24"
        self._eof = False
        self._ready = False

    def _open_control_media(self, value) -> bool:
        if not isinstance(value, dict):
            return False
        path = value.get("path")
        mode = str(value.get("mode", "")).strip().casefold()
        if not isinstance(path, str) or not path.strip() or mode not in {"video", "image", "file"}:
            return False
        if mode == "file":
            kind = media_kind_for_path(path)
            mode = "image" if kind == "image" else "video"
        try:
            position = float(value.get("position", 0.0))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(position) or position < 0.0:
            return False

        previous_path = self.path
        previous_mode = self.mode
        previous_position = self._current_position()
        previous_playing = bool(getattr(self, "playing", False))
        self.stop()
        self._reset_media_for_open(path.strip(), mode, position)
        try:
            self._open()
            if hasattr(self, "_browser"):
                self._browser.close()
                self._browser.add_recent(self.path)
            self._browser_mode = False
        except Exception as exc:
            requested_error = f"{type(exc).__name__}: {exc}"
            try:
                self.stop()
                self._reset_media_for_open(
                    previous_path, previous_mode, previous_position
                )
                self._open()
                if previous_mode == "video" and not previous_playing:
                    self._restart(previous_position, playing=False)
                self.error = requested_error
            except Exception as restore_exc:
                self.error = (
                    f"{requested_error}; previous media could not be restored: "
                    f"{type(restore_exc).__name__}: {restore_exc}"
                )
                self.playing = False
                self._ready = False
                self._position_anchor_time = None
            print(f"[media] {self.error}", flush=True)
        return True

    def _open_playlist_path(self, path: str) -> bool:
        media_mode = media_kind_for_path(path)
        if media_mode not in {"video", "image", "disc"}:
            return False
        opened = self._open_control_media({
            "path": path,
            "mode": "Image" if media_mode == "image" else "Video",
            "position": 0.0,
        })
        if opened:
            self._browser.close()
            self._browser_mode = False
        return opened

    def _read_control(self) -> None:
        paths = []
        for raw_path in (
            getattr(self.config, "media_control_path", None),
            getattr(self.config, "media_gui_control_path", None),
            getattr(self.config, "media_viewer_control_path", None),
        ):
            path = str(raw_path or "").strip()
            if path and path not in paths:
                paths.append(path)
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as stream:
                    request = json.load(stream)
            except (OSError, ValueError):
                continue
            if not isinstance(request, dict):
                continue
            if self._session_id and request.get("session") != self._session_id:
                continue
            try:
                revision = int(request.get("revision", -1))
            except (TypeError, ValueError):
                continue
            if revision <= self._last_control_revisions.get(path, -1):
                continue
            self._last_control_revisions[path] = revision
            action = str(request.get("action", "")).casefold()
            value = request.get("value")
            if action == "open":
                try:
                    if not self._open_control_media(value):
                        continue
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.playing = False
                    self._ready = False
                    print(f"[media] {self.error}", flush=True)
            elif action == "play" and not self._image:
                position = self._current_position()
                if getattr(self, "_eof", False) or (
                    self.duration > 0 and position >= self.duration - 0.05
                ):
                    position = 0.0
                self._browser_mode = False
                if not hasattr(self, "_frame_buffer_condition"):
                    self._restart(position, playing=True)
                elif self._has_open_video_decoder():
                    self._resume_playback(position)
                else:
                    self._restart(position, playing=True)
            elif action == "pause" and not self._image:
                if hasattr(self, "_frame_buffer_condition"):
                    self._pause_playback()
                else:
                    self._restart(self._current_position(), playing=False)
            elif action in {"seek", "seek_relative"} and not self._image:
                try:
                    amount = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(amount):
                    continue
                position = (
                    self._current_position() + amount
                    if action == "seek_relative"
                    else amount
                )
                if self.duration > 0:
                    position = min(position, self.duration)
                position = max(0.0, position)
                self._restart(position, playing=self.playing)
            elif action == "volume":
                try:
                    self.volume = max(0.0, min(1.0, float(value)))
                except (TypeError, ValueError):
                    continue
            elif action == "mute":
                self.muted = bool(value)
            elif action == "browse":
                self._browser_mode = True
                self._browser.open(value)
            elif action == "browse_up":
                self._browser.up()
            elif action == "browse_refresh":
                self._browser.refresh()
            elif action == "browse_close":
                self._browser.close()
                self._browser_mode = False
            elif action == "browse_select":
                path = value.get("path") if isinstance(value, dict) else value
                self._browser.select(str(path or ""))
            elif action == "browse_open":
                path = value.get("path") if isinstance(value, dict) else value
                kind, resolved = self._browser.open_path(str(path or ""))
                if kind == "media":
                    self._open_playlist_path(resolved)
                elif kind == "error":
                    self._browser.set_error(resolved)
            elif action == "recent_remove":
                path = value.get("path") if isinstance(value, dict) else value
                self._browser.remove_recent(str(path or ""))
            elif action in {"playlist_next", "playlist_previous"}:
                path = self._browser.playlist_step(
                    1 if action == "playlist_next" else -1
                )
                if path:
                    self._open_playlist_path(path)
            elif action == "playlist_select":
                path = value.get("path") if isinstance(value, dict) else value
                selected = self._browser.playlist_select(str(path or ""))
                if selected:
                    self._open_playlist_path(selected)
            elif action == "playlist_add":
                path = value.get("path") if isinstance(value, dict) else value
                self._browser.playlist_add(str(path or ""))
            elif action == "playlist_move" and isinstance(value, dict):
                try:
                    self._browser.playlist_move(str(value.get("path") or ""), int(value.get("offset", 0)))
                except (TypeError, ValueError):
                    continue
            elif action == "playlist_sort":
                descending = bool(value.get("descending", False)) if isinstance(value, dict) else False
                self._browser.playlist_sort(descending=descending)
            elif action == "playlist_page":
                try:
                    self._browser.playlist_page(int(value))
                except (TypeError, ValueError):
                    continue
            elif action == "playlist_clear":
                self._browser.playlist_clear()
            elif action == "playlist_remove":
                path = value.get("path") if isinstance(value, dict) else value
                self._browser.playlist_remove(str(path or ""))
            else:
                continue
            self._publish_state(force=True)

    def poll_control(self) -> None:
        with self._get_media_control_lock():
            self._read_control()
            self._publish_state()

    def _get_media_control_lock(self):
        lock = getattr(self, "_media_control_lock", None)
        if lock is None:
            lock = self._media_control_lock = threading.RLock()
        return lock

    def _read_d3d11_frame(self):
        decoder = self._native_decoder
        importer = self._native_frame_importer
        if decoder is None or importer is None:
            return None
        pumping = getattr(self, "_frame_pump_thread", None) is threading.current_thread()
        position = self._frame_pump_position if pumping else self._current_position()
        source_fps = float(getattr(self, "_source_fps", 0.0))
        playback_fps = float(getattr(self, "_playback_fps", 0.0) or 0.0)
        legacy_step = max(1, int(getattr(self, "_frame_pump_decimation_step", 1) or 1))
        selecting = pumping and source_fps > 0.0 and (
            (playback_fps > 0.0 and source_fps > playback_fps + 1e-6)
            or legacy_step > 1
        )
        advance = 1
        if selecting:
            if playback_fps > 0.0 and source_fps > playback_fps + 1e-6:
                advance = self._next_source_frame_advance() if self._source_frame_index > 0 else 1
            else:
                advance = legacy_step if self._source_frame_index > 0 else 1
            # Ask the decoder for an exact source slot, not wall-clock catch-up.
            # The half-source-period offset excludes the preceding frame whose
            # end timestamp equals the desired PTS. Discard before GPU VPP/HIP.
            position += (advance - 0.5) / source_fps
        try:
            frame = decoder.next_frame(position)
            if frame is None:
                return None
            tensor = importer.import_frame(frame)
            if selecting:
                self._decoder_resampled = True
                self._decoder_frame_advance = advance
            self._native_frame_pts = float(frame.pts_seconds)
            self._native_frame_duration = max(
                0.0, float(frame.duration_seconds)
            )
            self._frame = tensor
            return tensor
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            self.decode_fallback_reason = fallback_reason
            print(
                f"[media] D3D11VA GPU decode failed ({fallback_reason}); "
                "falling back to FFmpeg",
                flush=True,
            )
            self._close_native_decoder()
            if self._amf_decoder and not self._amf_disabled:
                self._decode_backend = "ffmpeg-amf"
                self._start_ffmpeg_video(
                    position, realtime=True, decoder=self._amf_decoder
                )
            else:
                self._decode_backend = "ffmpeg-software"
                self._start_ffmpeg_video(position, realtime=True)
            self._frame_format = (
                getattr(self, "_ffmpeg_frame_format", "bgr24")
                if getattr(self, "_frame_pixel_format", "bgr24")
                in {"nv12", "p010le"}
                else "bgr24_sdr_bt709"
            )
            return self._read_video_frame()

    def _read_video_frame(self):
        if getattr(self, "_native_decoder", None) is not None:
            return self._read_d3d11_frame()
        if self._decode_backend.startswith("nvdec"):
            return self._read_nvdec_frame()
        if self._decode_backend == "opencv-software":
            return self._read_opencv_frame()
        process = self._video_process
        if process is None or process.stdout is None:
            return None
        pixel_format = getattr(self, "_frame_pixel_format", "bgr24")
        decode_width = int(getattr(self, "_decode_width", 0) or self.width)
        decode_height = int(getattr(self, "_decode_height", 0) or self.height)
        frame_bytes = (
            decode_width * decode_height * 3
            if pixel_format == "p010le"
            else decode_width * decode_height * 3 // 2
            if pixel_format == "nv12"
            else decode_width * decode_height * 3
        )
        if frame_bytes <= 0:
            return None
        frame_buffer = bytearray(frame_bytes)
        frame_view = memoryview(frame_buffer)
        bytes_read = 0
        while bytes_read < frame_bytes:
            count = process.stdout.readinto(frame_view[bytes_read:])
            if not count:
                break
            bytes_read += count
        if bytes_read != frame_bytes:
            error = self._read_process_error(process)
            return_code = process.poll()
            hardware_unavailable = bool(
                error and _is_hardware_decode_unavailable(RuntimeError(error))
            )
            software_decoder_unavailable = bool(
                error and _is_software_decoder_unavailable(RuntimeError(error))
            )
            amf_vpp_unavailable = bool(
                self._decode_backend == "ffmpeg-amf" and _is_amf_vpp_unavailable(error)
            )
            amf_process_crash = bool(
                self._decode_backend == "ffmpeg-amf"
                and return_code not in (None, 0)
                and not error
            )
            decoder_unavailable = (
                hardware_unavailable
                or software_decoder_unavailable
                or amf_vpp_unavailable
                or amf_process_crash
            )
            if decoder_unavailable or error or return_code not in (None, 0):
                self._terminate(process)
                self._video_process = None
                if decoder_unavailable:
                    if amf_process_crash:
                        self.decode_fallback_reason = (
                            f"FFmpeg AMF decoder exited with status {return_code}"
                        )
                    else:
                        self.decode_fallback_reason = (
                            f"FFmpeg decoder unavailable: {error[-1000:]}"
                        )
                    if self._decode_backend == "ffmpeg-amf":
                        self._amf_disabled = True
                        fallback_position = self._current_position()
                        restart_audio = self.playing and bool(
                            getattr(self, "_has_audio", False)
                        )
                        if restart_audio:
                            self._close_audio()
                        self._start_video(fallback_position, realtime=self.playing)
                        if restart_audio:
                            self._start_audio(fallback_position)
                        return self._read_video_frame()
                    elif self._nvdec_source_path:
                        try:
                            if getattr(self, "_frame", None) is not None:
                                fallback_position = self._current_position()
                            elif getattr(self, "_audio_started_at", None) is not None:
                                audio_anchor = float(
                                    getattr(
                                        self,
                                        "_audio_position_anchor",
                                        getattr(self, "_position_anchor", 0.0),
                                    )
                                )
                                fallback_position = audio_anchor + max(
                                    0.0, time.perf_counter() - self._audio_started_at
                                )
                            else:
                                fallback_position = float(
                                    getattr(
                                        self,
                                        "_position_anchor",
                                        getattr(self, "position", 0.0),
                                    )
                                )
                            restart_audio = self.playing and bool(
                                getattr(self, "_has_audio", False)
                            )
                            if restart_audio:
                                self._close_audio()
                            self._start_opencv_video(
                                fallback_position, realtime=self.playing
                            )
                            if restart_audio:
                                self._start_audio(fallback_position)
                            return self._read_opencv_frame()
                        except Exception as exc:
                            self._close_opencv()
                            self.error = f"CPU video decode fallback failed: {exc}"
                    else:
                        self.error = (
                            "CPU video decode fallback requires a single video file; "
                            "the FFmpeg decoder reported unsupported hardware"
                        )
                elif error:
                    self.error = f"FFmpeg video decode failed: {error[-1000:]}"
                else:
                    self.error = f"FFmpeg video decoder exited with status {return_code}"
            return None
        ffmpeg_step = max(1, int(getattr(self, "_ffmpeg_decimation_step", 1)))
        ffmpeg_selection = bool(getattr(self, "_ffmpeg_selection_active", False))
        if ffmpeg_selection:
            schedule = getattr(self, "_ffmpeg_selection_schedule", None)
            self._decoder_resampled = True
            self._decoder_frame_advance = (
                1
                if getattr(self, "_ffmpeg_first_frame_pending", False) or schedule is None
                else schedule.next_advance()
            )
            self._ffmpeg_first_frame_pending = False
        elif ffmpeg_step > 1:
            self._decoder_resampled = True
            self._decoder_frame_advance = (
                1 if getattr(self, "_ffmpeg_first_frame_pending", False) else ffmpeg_step
            )
            self._ffmpeg_first_frame_pending = False
        if pixel_format in {"nv12", "p010le"}:
            dtype = np.dtype("<u2") if pixel_format == "p010le" else np.uint8
            self._frame = np.frombuffer(frame_buffer, dtype=dtype).reshape(
                decode_height * 3 // 2, decode_width
            )
        else:
            self._frame = np.frombuffer(frame_buffer, dtype=np.uint8).reshape(
                decode_height, decode_width, 3
            )
        return self._frame

    @staticmethod
    def _read_process_error(process) -> str:
        stream = getattr(process, "_media_stderr", None)
        if stream is None:
            return ""
        try:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 8192), os.SEEK_SET)
            return stream.read().decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            return ""

    def _start_opencv_video(self, position: float, *, realtime: bool) -> None:
        if not self._nvdec_source_path:
            raise RuntimeError("CPU fallback is only available for a single video file")
        import cv2

        acceleration = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
        software = getattr(cv2, "VIDEO_ACCELERATION_NONE", None)
        if acceleration is None or software is None:
            raise RuntimeError("OpenCV cannot force CPU video decoding in this build")
        capture = cv2.VideoCapture(
            self._nvdec_source_path,
            cv2.CAP_FFMPEG,
            [acceleration, software],
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not open the video for CPU decoding")
        if position > 0 and not capture.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0):
            capture.release()
            raise RuntimeError("OpenCV could not seek to the requested playback position")
        self._opencv_capture = capture
        self._opencv_realtime = bool(realtime)
        self._opencv_next_frame_at = None
        self._opencv_skip_fraction = 0.0
        self._opencv_first_frame_pending = True
        self._decode_backend = "opencv-software"
        self._frame_pixel_format = "bgr24"
        if self.hdr:
            transfer = "pq" if self.transfer == "smpte2084" else "hlg"
            primaries = self.primaries or "bt2020"
            self._frame_format = f"bgr24_{transfer}_{primaries}"
        elif self.primaries == "bt2020":
            self._frame_format = "bgr24_sdr_bt2020"
        else:
            self._frame_format = "bgr24"

    def _read_opencv_frame(self):
        capture = self._opencv_capture
        if capture is None:
            return None
        deadline = time.perf_counter()
        interval = 1.0 / max(1.0, self._playback_fps)
        pumping = getattr(self, "_frame_pump_thread", None) is threading.current_thread()
        if self._opencv_realtime and not pumping:
            deadline = self._opencv_next_frame_at or deadline
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        skipped_frames = 0
        if getattr(self, "_opencv_first_frame_pending", False):
            self._opencv_first_frame_pending = False
        elif self._source_fps > self._playback_fps > 0:
            self._opencv_skip_fraction += self._source_fps / self._playback_fps - 1.0
            while self._opencv_skip_fraction >= 1.0:
                if not capture.grab():
                    return None
                self._opencv_skip_fraction -= 1.0
                skipped_frames += 1
        ok, frame = capture.read()
        if not ok:
            return None
        if self._opencv_realtime:
            self._opencv_next_frame_at = max(deadline + interval, time.perf_counter())
        self._decoder_resampled = True
        self._decoder_frame_advance = 1 + skipped_frames
        self._frame = frame
        return frame

    def _close_opencv(self) -> None:
        capture = getattr(self, "_opencv_capture", None)
        self._opencv_capture = None
        self._opencv_next_frame_at = None
        if capture is not None:
            capture.release()

    def _read_nvdec_frame(self):
        if self._nvdec_first_frame_pending:
            self._nvdec_first_frame_pending = False
            self._nvdec_next_frame_at = time.perf_counter() + 1.0 / max(1.0, self._playback_fps)
            return self._frame
        interval = 1.0 / max(1.0, self._playback_fps)
        deadline = self._nvdec_next_frame_at or time.perf_counter()
        delay = deadline - time.perf_counter()
        pumping = getattr(self, "_frame_pump_thread", None) is threading.current_thread()
        if delay > 0 and not pumping:
            time.sleep(delay)
        try:
            frame_count = 1
            if self._source_fps > self._playback_fps > 0:
                self._nvdec_skip_fraction += self._source_fps / self._playback_fps - 1.0
                skipped_frames = int(self._nvdec_skip_fraction)
                frame_count += skipped_frames
                self._nvdec_skip_fraction -= skipped_frames
            frame = (
                self._decode_nvdec_frame()
                if frame_count == 1
                else self._decode_nvdec_frames(frame_count)
            )
            if frame is None:
                return None
        except Exception as exc:
            if not _is_hardware_decode_unavailable(exc):
                raise
            self._nvdec_disabled = True
            self.decode_fallback_reason = f"{type(exc).__name__}: {exc}"
            position = self._current_position()
            print(
                f"[media] NVDEC became unavailable ({self.decode_fallback_reason}); using FFmpeg software decoding",
                flush=True,
            )
            restart_audio = self.playing and bool(getattr(self, "_has_audio", False))
            if restart_audio:
                self._close_audio()
            self._start_video(position, realtime=True)
            if restart_audio:
                self._start_audio(position)
            return self._read_video_frame()
        self._decoder_frame_advance = max(1, int(getattr(self, "_nvdec_last_frame_advance", frame_count)))
        self._decoder_terminal_partial = self._decoder_frame_advance < frame_count
        self._decoder_resampled = True
        self._nvdec_next_frame_at = max(deadline + interval, time.perf_counter())
        self._frame = frame
        return frame

    def _publish_state(self, *, force: bool = False) -> None:
        path = str(self.config.media_state_path or "").strip()
        now = time.monotonic()
        if not path or (not force and now - self._last_publish_at < 1.0):
            return
        self._last_publish_at = now
        frame = (
            self._displayed_frame
            if getattr(self, "_displayed_frame", None) is not None
            else getattr(self, "_frame", None)
        )
        frame_device = str(getattr(getattr(frame, "device", None), "type", "") or "")
        if hasattr(self, "_frame_buffer_condition"):
            with self._frame_buffer_condition:
                buffered_frames = len(self._frame_buffer)
            frame_buffer_capacity = self._frame_buffer_capacity
        else:
            buffered_frames = 0
            frame_buffer_capacity = 0
        data = {
            "pid": os.getpid(),
            "session": self._session_id,
            "path": self.path,
            "mode": (
                "Browser" if getattr(self, "_browser_mode", False)
                else "Image" if self._image
                else "Video"
            ),
            "position": self._current_position(),
            "duration": self.duration,
            "audio_sync": self.audio_sync_metrics(),
            "playing": self.playing,
            "ready": bool(getattr(self, "_ready", False)),
            "volume": self.volume,
            "muted": self.muted,
            "hdr": self.hdr,
            "gpu_resident": frame_device in {"cuda", "mps", "xpu", "hip"},
            "buffered_frames": buffered_frames,
            "frame_buffer_capacity": frame_buffer_capacity,
            "playback_fps": getattr(self, "_playback_fps", 0.0),
            "source_fps": getattr(self, "_source_fps", 0.0),
            "cadence_mode": "adaptive",
            "cadence_reason": getattr(getattr(self, "_media_cadence_controller", None), "reason", "warming_up"),
            "decode_backend": self._decode_backend,
            "decode_fallback_reason": self.decode_fallback_reason,
            "audio_enabled": bool(getattr(self, "_has_audio", False)),
            "error": self.error,
            "audio_error": self.audio_error,
            **(self._browser.snapshot() if hasattr(self, "_browser") else {}),
        }
        temp_path = f"{path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False)
            for attempt in range(5):
                try:
                    os.replace(temp_path, path)
                    if self.error.startswith("Could not save media state:"):
                        self.error = ""
                    break
                except OSError as exc:
                    if not isinstance(exc, PermissionError) or attempt == 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        except OSError as exc:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            self.error = f"Could not save media state: {exc}"

    def _finish_video_stream(self):
        if hasattr(self, "_frame_buffer_condition"):
            self._stop_frame_pump()
        self._close_native_decoder()
        self._terminate(self._video_process)
        self._video_process = None
        self._close_nvdec()
        self._close_opencv()
        self._close_audio()
        next_path = (
            self._browser.playlist_step(1)
            if hasattr(self, "_browser")
            else None
        )
        if next_path and self._open_playlist_path(next_path):
            next_frame, _next_size = self.grab()
            return next_frame
        final_position = self._current_position()
        self.playing = False
        self.position = self.duration or final_position
        self._eof = True
        self._position_anchor = self.position
        self._position_anchor_time = None
        # Keep the final displayed frame visible after EOF instead of returning
        # one black frame and stalling the presenter.
        frame = (
            self._displayed_frame
            if getattr(self, "_displayed_frame", None) is not None
            else getattr(self, "_frame", None)
        )
        self._publish_state(force=True)
        return frame

    def grab(self):
        with self._get_media_control_lock():
            return self._grab_media_frame()

    def _grab_media_frame(self):
        self._read_control()
        if not self._ready:
            time.sleep(0.1)
            self._publish_state()
            return None, (self.width, self.height)
        if getattr(self, "_browser_mode", False):
            self._current_source_frame_identity = None
            now = time.monotonic()
            if self._frame is None or now - self._last_browser_frame_at >= 0.2:
                width, height = _browser_output_size(
                    self.config, self.width, self.height
                )
                self.width, self.height = width, height
                self._frame = np.full((height, width, 3), 8, dtype=np.uint8)
                self._frame_format = "bgr24"
                self._last_browser_frame_at = now
            else:
                time.sleep(0.01)
            self._publish_state()
            return self._frame, (self.width, self.height)
        frame = None
        if self._image:
            now = time.monotonic()
            if self._frame is not None and now - self._last_image_frame_at >= 1.0:
                self._last_image_frame_at = now
                frame = self._frame
            else:
                time.sleep(0.05)
                frame = None
        elif self.playing:
            if not hasattr(self, "_frame_buffer_condition"):
                frame = self._read_video_frame()
                if frame is None:
                    frame = self._finish_video_stream()
                else:
                    source_index = int(getattr(self, "_source_frame_index", 0))
                    self._source_frame_index = source_index + 1
                    self._current_source_frame_identity = (
                        self.media_generation,
                        source_index,
                        float(self._current_position()),
                    )
            else:
                if self._frame_pump_thread is None or not self._frame_pump_thread.is_alive():
                    self._start_frame_pump()
                if (getattr(self, "_presentation_clock_pending", False)
                        and getattr(self, "_presentation_first_frame_taken", False)):
                    with self._frame_buffer_condition:
                        self._frame_buffer_condition.wait(timeout=0.02)
                    return None, (self.width, self.height)
                first_frame = self._position_anchor_time is None
                item = self._take_buffered_frame(
                    self._current_position() + float(getattr(self, "_media_prepare_lead_s", 0.0)),
                    allow_first=first_frame,
                )
                if item is not None:
                    source_pts, frame, source_index = item[:3]
                    self._current_source_metadata = dict(item[3]) if len(item) > 3 else {}
                    if first_frame:
                        if getattr(self, "_resume_clock_at", None) is None:
                            self._position_anchor = float(source_pts)
                        if getattr(self, "_presentation_start_index", None) is None:
                            self._presentation_start_index = int(source_index)
                        self._presentation_first_frame_taken = True
                        if not getattr(self, "_presentation_clock_pending", False):
                            self._position_anchor_time = time.perf_counter()
                    self._current_source_frame_identity = (
                        self.media_generation,
                        int(source_index),
                        float(source_pts),
                    )
                    self._current_source_metadata["media_presentation_clock_pending"] = bool(
                        getattr(self, "_presentation_clock_pending", False)
                    )
                    anchor_at = self._position_anchor_time
                    self._current_source_metadata["media_presentation_deadline"] = (
                        float(anchor_at) + float(source_pts) - float(self._position_anchor)
                        if anchor_at is not None and bool(getattr(self.config, "media_presentation_sync", False))
                        else None
                    )
                    if str(getattr(getattr(frame, "device", None), "type", "")) == "cuda":
                        import torch

                        frame.record_stream(torch.cuda.current_stream(frame.device))
                    self._displayed_frame = frame
                    self._frame = frame
                else:
                    with self._frame_buffer_condition:
                        pump_error = self._frame_pump_error
                        pump_done = self._frame_pump_done
                        buffer_empty = not self._frame_buffer
                    if pump_error:
                        self.error = f"Media frame pump failed: {pump_error}"
                        print(f"[media] {self.error}", flush=True)
                    if pump_done and buffer_empty:
                        if self.duration > 0.0 and self._current_position() < self.duration:
                            # Decode-ahead/prepare lead may reach EOF before
                            # the last visible frame's source-time interval ends.
                            with self._frame_buffer_condition:
                                self._frame_buffer_condition.wait(timeout=0.01)
                        else:
                            frame = self._finish_video_stream()
        else:
            time.sleep(0.05)
            displayed_frame = getattr(self, "_displayed_frame", None)
            frame = displayed_frame if displayed_frame is not None else self._frame
        self._publish_state()
        return frame, (self.width, self.height)

    def stop(self) -> None:
        self.position = self._current_position()
        if hasattr(self, "_frame_buffer_condition"):
            self._stop_frame_pump()
        self._close_native_decoder()
        self._terminate(self._video_process)
        self._video_process = None
        self._close_nvdec()
        self._close_opencv()
        self._close_audio()
        if hasattr(self, "_frame_buffer_condition"):
            self._reset_frame_buffer(self.position)
        self._displayed_frame = None
        if hasattr(self, "_browser"):
            self._browser.close()
        self.playing = False
        self._ready = False
        self._position_anchor = self.position
        self._position_anchor_time = None
        self._publish_state(force=True)
        if self._manifest_path:
            try:
                os.unlink(self._manifest_path)
            except FileNotFoundError:
                pass
            self._manifest_path = None
        if self._owns_iso_mount and self._mounted_iso_path:
            try:
                _dismount_iso(self._mounted_iso_path)
            except Exception as exc:
                self.error = f"Could not dismount ISO: {exc}"
                print(f"[media] {self.error}", flush=True)
            self._owns_iso_mount = False
            self._publish_state(force=True)
