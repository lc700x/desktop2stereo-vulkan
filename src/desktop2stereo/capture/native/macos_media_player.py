"""Native AVFoundation/VideoToolbox playback with retained GPU frames."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

_NATIVE_DIR = Path(__file__).resolve().parent
_SOURCE = _NATIVE_DIR / "macos_media_player.mm"
_HEADER = _NATIVE_DIR / "macos_media_player.h"
_LIBRARY = None
_LIBRARY_LOCK = threading.Lock()
_MATROSKA_SUFFIXES = {".mkv", ".mk3d"}


def _ffmpeg_binary() -> str | None:
    configured = os.environ.get("D2S_FFMPEG_PATH", "").strip()
    candidates = [Path(configured)] if configured else []
    runtime_root = Path(
        os.environ.get(
            "D2S_STREAMING_RUNTIME_DIR",
            Path(__file__).resolve().parents[2] / "streaming" / "rtmp",
        )
    )
    executable = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates.append(runtime_root / "ffmpeg" / "bin" / executable)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which(executable)


def _prepare_media_asset(path: str | os.PathLike[str]) -> tuple[str, str | None]:
    """Make Matroska readable by AVFoundation without decoding video pixels."""
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() not in _MATROSKA_SUFFIXES:
        return str(source), None
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError(
            "MKV playback requires the bundled FFmpeg remuxer; no ffmpeg executable was found"
        )
    temporary_dir = tempfile.mkdtemp(prefix="d2s-macos-mkv-")
    destination = Path(temporary_dir) / f"{source.stem}.mp4"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise RuntimeError(f"MKV container remux failed: {exc}") from exc
    if result.returncode != 0 or not destination.is_file():
        detail = (result.stderr or "").strip() or "FFmpeg produced no MP4 output"
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise RuntimeError(f"MKV container remux failed: {detail}")
    return str(destination), temporary_dir


def _cleanup_media_asset(temporary_dir: str | None) -> None:
    if temporary_dir:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def _cache_path() -> Path:
    override = os.environ.get("D2S_MEDIA_NATIVE_CACHE")
    root = (
        Path(override).expanduser()
        if override
        else Path.home() / "Library" / "Caches" / "desktop2stereo" / "media"
    )
    digest = hashlib.sha256(_SOURCE.read_bytes() + _HEADER.read_bytes()).hexdigest()[:16]
    return root / f"libd2s_media-{platform.machine()}-{digest}.dylib"


def _build_library() -> tuple[Path, str | None]:
    if platform.system() != "Darwin":
        raise RuntimeError("native AVFoundation media playback is available only on macOS")
    clang = shutil.which("clang++")
    if clang is None:
        raise RuntimeError("clang++ is unavailable for the native AVFoundation media bridge")
    target = _cache_path()
    if target.is_file():
        return target, None
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d2s-media-") as temporary_dir:
        temporary = Path(temporary_dir) / target.name
        command = [
            clang,
            "-std=c++11",
            "-fobjc-arc",
            "-ObjC++",
            "-dynamiclib",
            "-O3",
            "-mmacosx-version-min=13.0",
            str(_SOURCE),
            "-framework",
            "AVFoundation",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreVideo",
            "-framework",
            "VideoToolbox",
            "-framework",
            "Foundation",
            "-o",
            str(temporary),
        ]
        try:
            sdk = subprocess.run(
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            sdk = ""
        if sdk:
            command[command.index(str(_SOURCE)) : command.index(str(_SOURCE))] = [
                "-isysroot",
                sdk,
            ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"native AVFoundation media bridge build failed: {result.stderr.strip()}"
            )
        if not temporary.is_file():
            raise RuntimeError("native AVFoundation media bridge produced no dylib")
        os.replace(temporary, target)
    return target, None


def _load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        path, _reason = _build_library()
        library = ctypes.CDLL(str(path))
        library.d2s_media_player_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.d2s_media_player_create.restype = ctypes.c_void_p
        library.d2s_media_player_copy_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_double),
        ]
        library.d2s_media_player_copy_frame.restype = ctypes.c_int32
        library.d2s_media_player_release_frame.argtypes = [ctypes.c_void_p]
        library.d2s_media_player_set_playing.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        library.d2s_media_player_set_playing.restype = ctypes.c_int32
        library.d2s_media_player_seek.argtypes = [ctypes.c_void_p, ctypes.c_double]
        library.d2s_media_player_seek.restype = ctypes.c_int32
        library.d2s_media_player_set_audio.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_int32,
        ]
        library.d2s_media_player_set_audio.restype = ctypes.c_int32
        library.d2s_media_player_position.argtypes = [ctypes.c_void_p]
        library.d2s_media_player_position.restype = ctypes.c_double
        library.d2s_media_player_duration.argtypes = [ctypes.c_void_p]
        library.d2s_media_player_duration.restype = ctypes.c_double
        library.d2s_media_player_status.argtypes = [ctypes.c_void_p]
        library.d2s_media_player_status.restype = ctypes.c_int32
        library.d2s_media_player_last_error.argtypes = [ctypes.c_void_p]
        library.d2s_media_player_last_error.restype = ctypes.c_char_p
        library.d2s_media_player_destroy.argtypes = [ctypes.c_void_p]
        library.d2s_media_player_destroy.restype = None
        _LIBRARY = library
        return library


class NativeMediaFrame:
    """Retained AVFoundation pixel buffer held until the native consumer releases it."""

    resource_kind = "avfoundation_cv_pixel_buffer"
    resource_format = "P010_10Bit"

    def __init__(self, library, pixel_buffer: int, presentation_time: float, hdr: bool):
        self._library = library
        self._pixel_buffer = ctypes.c_void_p(pixel_buffer)
        self.pixel_buffer_pointer = int(pixel_buffer)
        self.presentation_time = float(presentation_time)
        self.callback_time = None
        self.frame_status = "native_avplayer_output"
        self.format = "P010_10Bit" if hdr else "BGRA8Unorm"
        self.resource_format = self.format

    def release(self) -> None:
        pixel_buffer, self._pixel_buffer = self._pixel_buffer, None
        self.pixel_buffer_pointer = 0
        if pixel_buffer:
            self._library.d2s_media_player_release_frame(pixel_buffer)

    def __del__(self):
        self.release()


class MacOSMediaPlayer:
    def __init__(self, path: str, *, hdr: bool, audio_enabled: bool):
        self._handle = None
        self._temporary_asset_dir = None
        self._library = _load_library()
        error = ctypes.create_string_buffer(1024)
        asset_path, self._temporary_asset_dir = _prepare_media_asset(path)
        handle = self._library.d2s_media_player_create(
            asset_path.encode("utf-8"),
            int(hdr),
            int(audio_enabled),
            error,
            len(error),
        )
        if not handle:
            message = error.value.decode(errors="replace")
            _cleanup_media_asset(self._temporary_asset_dir)
            self._temporary_asset_dir = None
            raise RuntimeError(message or "AVFoundation media player creation failed")
        self._handle = ctypes.c_void_p(handle)
        self.hdr = bool(hdr)

    def copy_frame(self) -> NativeMediaFrame | None:
        handle = self._require_handle()
        pixel_buffer = ctypes.c_void_p()
        presentation_time = ctypes.c_double()
        result = self._library.d2s_media_player_copy_frame(
            handle, ctypes.byref(pixel_buffer), ctypes.byref(presentation_time)
        )
        if result == 0:
            return None
        if result < 0:
            raw_error = self._library.d2s_media_player_last_error(handle)
            message = raw_error.decode(errors="replace") if raw_error else "AVPlayer frame failed"
            raise RuntimeError(message)
        return NativeMediaFrame(
            self._library,
            int(pixel_buffer.value),
            float(presentation_time.value),
            self.hdr,
        )

    @property
    def position(self) -> float:
        return float(self._library.d2s_media_player_position(self._require_handle()))

    @property
    def duration(self) -> float:
        return float(self._library.d2s_media_player_duration(self._require_handle()))

    @property
    def status(self) -> int:
        return int(self._library.d2s_media_player_status(self._require_handle()))

    def set_playing(self, playing: bool) -> None:
        self._check(self._library.d2s_media_player_set_playing(
            self._require_handle(), int(bool(playing))
        ))

    def seek(self, position: float) -> None:
        self._check(self._library.d2s_media_player_seek(
            self._require_handle(), float(position)
        ))

    def set_audio(self, volume: float, muted: bool) -> None:
        self._check(self._library.d2s_media_player_set_audio(
            self._require_handle(), float(volume), int(bool(muted))
        ))

    def _require_handle(self):
        if not self._handle:
            raise RuntimeError("AVFoundation media player is closed")
        return self._handle

    @staticmethod
    def _check(result: int) -> None:
        if result != 0:
            raise RuntimeError("AVFoundation media player control failed")

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._library.d2s_media_player_destroy(handle)
        temporary_dir, self._temporary_asset_dir = self._temporary_asset_dir, None
        _cleanup_media_asset(temporary_dir)

    def __del__(self):
        self.close()
