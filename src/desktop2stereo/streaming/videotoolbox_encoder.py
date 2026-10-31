"""Hardware VideoToolbox encoder for native macOS Metal stream frames."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any


_SOURCE = Path(__file__).with_suffix(".mm")


def _native_library() -> ctypes.CDLL:
    if platform.system() != "Darwin":
        raise RuntimeError("Apple VideoToolbox encoding is available only on macOS")
    override = str(os.environ.get("D2S_VIDEOTOOLBOX_ENCODER_DYLIB", "") or "").strip()
    bundled = (
        Path(__file__).resolve().parent
        / "moonlight_server"
        / "macos"
        / "d2s_videotoolbox_encoder.dylib"
    )
    if override:
        library_path = Path(override).expanduser()
    elif bundled.is_file():
        library_path = bundled
    else:
        clang = shutil.which("clang++")
        if clang is None:
            raise RuntimeError("Xcode Command Line Tools are required for VideoToolbox encoding")
        digest = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()[:16]
        cache = Path.home() / "Library" / "Caches" / "Desktop2Stereo" / "native"
        cache.mkdir(parents=True, exist_ok=True)
        library_path = cache / f"videotoolbox-{digest}.dylib"
    if override and not library_path.is_file():
        raise RuntimeError(f"VideoToolbox bridge library is missing: {library_path}")
    if not library_path.is_file() and library_path != bundled:
        command = [
            clang,
            "-std=c++17",
            "-fobjc-arc",
            "-O2",
            "-dynamiclib",
            "-arch",
            "arm64",
            "-mmacosx-version-min=13.0",
            str(_SOURCE),
            "-framework",
            "Foundation",
            "-framework",
            "CoreMedia",
            "-framework",
            "CoreVideo",
            "-framework",
            "Metal",
            "-framework",
            "VideoToolbox",
            "-o",
        ]
        with tempfile.NamedTemporaryFile(
            dir=cache, prefix="videotoolbox-", suffix=".dylib", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            result = subprocess.run(
                [*command, str(temporary_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    "Could not build the VideoToolbox bridge: "
                    + (result.stderr.strip() or result.stdout.strip())
                )
            os.replace(temporary_path, library_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    if not library_path.is_file():
        raise RuntimeError(f"VideoToolbox bridge library is missing: {library_path}")
    library = ctypes.CDLL(str(library_path))
    library.d2s_videotoolbox_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.d2s_videotoolbox_create.restype = ctypes.c_void_p
    library.d2s_videotoolbox_source_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.d2s_videotoolbox_source_buffer.restype = ctypes.c_void_p
    library.d2s_videotoolbox_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.d2s_videotoolbox_encode.restype = ctypes.c_int
    library.d2s_videotoolbox_destroy.argtypes = [ctypes.c_void_p]
    library.d2s_videotoolbox_destroy.restype = None
    return library


class VideoToolboxEncoder:
    """Encode packed native CoreML/Metal frames as Annex-B access units."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        *,
        hevc: bool,
        fit_mode: str = "contain",
        display_mode: str = "Half-SBS",
        input_size: tuple[int, int] | None = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.hevc = bool(hevc)
        self.fit_mode = str(fit_mode or "contain")
        self.display_mode = str(display_mode or "Half-SBS")
        self.input_size = input_size
        self._library = _native_library()
        error = ctypes.create_string_buffer(512)
        self._handle = self._library.d2s_videotoolbox_create(
            self.width,
            self.height,
            self.fps,
            max(64_000, int(bitrate)),
            int(self.hevc),
            error,
            len(error),
        )
        if not self._handle:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
    @staticmethod
    def _packed_size(source: tuple[int, int], output_format: str) -> tuple[int, int]:
        width, height = source
        normalized = str(output_format).strip().casefold().replace("-", "_")
        if normalized == "full_sbs":
            return width * 2, height
        if normalized == "full_tab":
            return width, height * 2
        return width, height

    def encode_native_frame(
        self,
        native_frame: Any,
        *,
        output_format: str,
        frame_index: int,
        force_keyframe: bool,
    ) -> dict[str, Any]:
        if not self._handle:
            raise RuntimeError("VideoToolbox encoder is closed")
        from streaming.aspect import presentation_blit_regions

        source_size = tuple(map(int, native_frame.source_size))
        packed_size = self._packed_size(source_size, output_format)
        normalized_output = str(output_format).strip().casefold().replace("-", "_")
        if normalized_output == "half_sbs":
            warp_config = dict(getattr(native_frame, "warp_config", None) or {})
            warp_config["half_sbs_single_sample"] = 1
            warp_config["stream_bgra"] = 1
            configure_warp = getattr(native_frame, "configure_warp", None)
            if callable(configure_warp):
                configure_warp(**warp_config)
            else:
                try:
                    native_frame.warp_config = warp_config
                except (AttributeError, TypeError):
                    pass
        regions = presentation_blit_regions(
            packed_size,
            (self.width, self.height),
            self.fit_mode,
            self.display_mode,
            self.input_size,
        )
        if not regions or len(regions) > 2:
            raise RuntimeError("Could not fit the stereo frame to Moonlight's negotiated size")
        flattened_regions = []
        for source_rect, target_rect in regions:
            flattened_regions.extend((*source_rect, *target_rect))
        flattened_regions.extend([0] * ((2 - len(regions)) * 8))
        region_array = (ctypes.c_int32 * 16)(*flattened_regions)
        error = ctypes.create_string_buffer(512)
        source_buffer = self._library.d2s_videotoolbox_source_buffer(
            self._handle,
            packed_size[0],
            packed_size[1],
            error,
            len(error),
        )
        if not source_buffer:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        native_frame.pack_metal(
            source_buffer,
            packed_size[0] * packed_size[1] * 4,
            packed_size,
            output_format,
        )
        output = ctypes.POINTER(ctypes.c_uint8)()
        output_size = ctypes.c_size_t()
        keyframe = ctypes.c_int()
        error = ctypes.create_string_buffer(512)
        result = self._library.d2s_videotoolbox_encode(
            self._handle,
            int(frame_index),
            int(bool(force_keyframe)),
            len(regions),
            int(normalized_output == "half_sbs"),
            region_array,
            ctypes.byref(output),
            ctypes.byref(output_size),
            ctypes.byref(keyframe),
            error,
            len(error),
        )
        if result:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        return {
            "data": ctypes.string_at(output, output_size.value),
            "frame_index": int(frame_index),
            "idr": bool(keyframe.value),
        }

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        self._handle = None
        if handle:
            self._library.d2s_videotoolbox_destroy(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
