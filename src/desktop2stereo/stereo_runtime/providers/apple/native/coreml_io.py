"""Lazy Objective-C Core ML/Metal bridge for the macOS local viewer.

The bridge is deliberately optional. It is built into the user cache only on
Darwin, and every capability failure returns control to the existing Python
Core ML path rather than changing other providers.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Any

from ....output import output_edge_aa_enabled


_NATIVE_DIR = Path(__file__).resolve().parent
_SOURCE = _NATIVE_DIR / "macos_coreml_io.mm"
_HEADER = _NATIVE_DIR / "macos_coreml_io.h"
_FXAA_HEADER = _NATIVE_DIR / "sbs_fxaa_msl.h"
_LOGGER = logging.getLogger(__name__)

_NATIVE_OUTPUT_FORMATS = {
    "half_sbs": 0,
    "full_sbs": 1,
    "half_tab": 2,
    "full_tab": 3,
    "mono": 4,
    "depth_map": 5,
    "anaglyph": 6,
    "interleaved": 7,
    "leia": 8,
}
_NATIVE_ANAGLYPH_METHODS = {
    "red_cyan": 0,
    "green_magenta": 1,
    "amber_blue": 2,
    "gray": 3,
}


class _NativeResult(ctypes.Structure):
    _fields_ = [
        ("slot", ctypes.c_int32),
        ("source_width", ctypes.c_int32),
        ("source_height", ctypes.c_int32),
        ("input_width", ctypes.c_int32),
        ("input_height", ctypes.c_int32),
        ("depth_width", ctypes.c_int32),
        ("depth_height", ctypes.c_int32),
        ("input_shared", ctypes.c_int32),
        ("output_backing_used", ctypes.c_int32),
        ("output_zero_copy", ctypes.c_int32),
        ("finite_depth", ctypes.c_int32),
        ("nonfinite_count", ctypes.c_uint32),
        ("normalize_lo", ctypes.c_float),
        ("normalize_hi", ctypes.c_float),
        ("preprocess_ms", ctypes.c_double),
        ("model_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double),
        ("raw_normalize_lo", ctypes.c_float),
        ("raw_normalize_hi", ctypes.c_float),
        ("normalization_history_reset", ctypes.c_int32),
    ]


class _NativeWarpConfig(ctypes.Structure):
    _fields_ = [
        ("depth_strength", ctypes.c_float),
        ("max_disparity_px", ctypes.c_float),
        ("convergence", ctypes.c_float),
        ("edge_threshold", ctypes.c_float),
        ("fill_strength", ctypes.c_float),
        ("fill_radius", ctypes.c_int32),
        ("mask_feather_radius", ctypes.c_int32),
        ("symmetric", ctypes.c_int32),
        ("layers", ctypes.c_int32),
        ("softness", ctypes.c_float),
        ("foreground_scale", ctypes.c_float),
        ("midground_scale", ctypes.c_float),
        ("background_scale", ctypes.c_float),
        ("edge_dilation", ctypes.c_int32),
        ("screen_edge_suppression", ctypes.c_int32),
        ("hole_fill_mode", ctypes.c_int32),
        ("occlusion_enabled", ctypes.c_int32),
        ("depth_pop", ctypes.c_float),
        ("antialias_strength", ctypes.c_float),
        ("anaglyph_method", ctypes.c_int32),
        ("edge_aa_enabled", ctypes.c_int32),
    ]


class NativeCoreMLBusy(RuntimeError):
    """The bounded native resource ring has no reusable slot yet."""


def native_output_format_id(output_format: str) -> int:
    try:
        return _NATIVE_OUTPUT_FORMATS[str(output_format).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"native CoreML output format is unsupported: {output_format}") from exc


def native_anaglyph_method_id(method: str) -> int:
    try:
        return _NATIVE_ANAGLYPH_METHODS[str(method).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"native CoreML anaglyph method is unsupported: {method}") from exc


def native_io_enabled() -> bool:
    return (
        os.name == "posix"
        and platform.system() == "Darwin"
        and os.environ.get("D2S_COREML_NATIVE_IO", "1").strip().lower()
        not in {"0", "false", "off"}
    )


def _cache_path() -> Path:
    override = os.environ.get("D2S_COREML_NATIVE_IO_CACHE")
    root = (
        Path(override).expanduser()
        if override
        else Path.home() / "Library" / "Caches" / "desktop2stereo" / "coreml_io"
    )
    digest = hashlib.sha256(
        _SOURCE.read_bytes() + _HEADER.read_bytes() + _FXAA_HEADER.read_bytes()
    ).hexdigest()[:16]
    return root / f"libd2s_coreml_io-{platform.machine()}-{digest}.dylib"


def _sdk_path() -> str | None:
    try:
        result = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _build_library() -> tuple[Path | None, str | None]:
    if not all(path.is_file() for path in (_SOURCE, _HEADER, _FXAA_HEADER)):
        return None, "native CoreML bridge source is missing"
    clang = shutil.which("clang++")
    if clang is None:
        return None, "clang++ is unavailable"
    target = _cache_path()
    if target.is_file():
        return target, None
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d2s-coreml-io-") as temp_dir:
        temporary = Path(temp_dir) / target.name
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
            "Foundation",
            "-framework",
            "CoreML",
            "-framework",
            "CoreVideo",
            "-framework",
            "Metal",
            "-framework",
            "IOSurface",
            "-o",
            str(temporary),
        ]
        sdk = _sdk_path()
        if sdk:
            command[command.index(str(_SOURCE)) : command.index(str(_SOURCE))] = [
                "-isysroot",
                sdk,
            ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            return None, f"native CoreML bridge build failed: {detail.strip()}"
        if not temporary.is_file():
            return None, "native CoreML bridge build produced no dylib"
        os.replace(temporary, target)
    return target, None


def _buffer_pointer(value: Any) -> tuple[ctypes.c_void_p, memoryview]:
    view = memoryview(value)
    if view.readonly:
        raise TypeError("native Vulkan destination must be writable")
    if not view.contiguous:
        view = view.cast("B")
    if view.format != "B":
        view = view.cast("B")
    pointer = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
    return ctypes.c_void_p(pointer), view


@dataclass
class NativeCoreMLFrame:
    """A native depth result retained by one bridge resource-ring slot."""

    bridge: "NativeCoreMLIOBridge"
    slot: int
    frame_id: int
    source_width: int
    source_height: int
    depth_width: int
    depth_height: int
    finite_depth: bool
    nonfinite_count: int
    preprocess_ms: float
    model_ms: float
    postprocess_ms: float
    input_shared: bool
    output_backing_used: bool
    output_zero_copy: bool
    normalize_lo: float = 0.0
    normalize_hi: float = 1.0
    raw_normalize_lo: float = 0.0
    raw_normalize_hi: float = 1.0
    normalization_history_reset: bool = False
    warp_config: dict[str, float | int | str] | None = None
    released: bool = False

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (1, 1, int(self.depth_height), int(self.depth_width))

    @property
    def source_size(self) -> tuple[int, int]:
        return (int(self.source_width), int(self.source_height))

    def pack(
        self,
        destination: Any,
        output_size: tuple[int, int],
        output_format: str,
        *,
        rgb: bool = False,
    ) -> None:
        if self.released:
            raise RuntimeError("native CoreML frame was already released")
        pointer, view = _buffer_pointer(destination)
        width, height = (int(output_size[0]), int(output_size[1]))
        arguments = (self, pointer, len(view), width, height, output_format)
        if rgb:
            self.bridge.pack(*arguments, rgb=True)
        else:
            self.bridge.pack(*arguments)

    def configure_warp(self, **values: float | int | str) -> None:
        """Attach the shared stereo parameters before presenter-side packing."""
        self.warp_config = dict(values)

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.bridge.release(self.slot)


class NativeCoreMLIOBridge:
    """ctypes owner for the native Core ML/Metal resource ring."""

    def __init__(self, model_path: str | Path, input_width: int, input_height: int):
        if not native_io_enabled():
            raise RuntimeError("native CoreML IO is disabled")
        library_path, build_error = _build_library()
        if library_path is None:
            raise RuntimeError(build_error or "native CoreML bridge unavailable")
        self._library = ctypes.CDLL(str(library_path))
        # Keep the C context alive while calls or native frames are outstanding,
        # but do not serialize prediction and packing. The native ring is the
        # synchronization boundary for those independent operations.
        self._state_condition = threading.Condition()
        self._active_calls = 0
        self._outstanding_frames = 0
        self._closing = False
        self._library.d2s_coreml_io_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._library.d2s_coreml_io_create.restype = ctypes.c_void_p
        self._library.d2s_coreml_io_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_NativeResult),
        ]
        self._library.d2s_coreml_io_predict.restype = ctypes.c_int32
        pack_argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(_NativeWarpConfig),
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._library.d2s_coreml_io_pack.argtypes = pack_argtypes
        self._library.d2s_coreml_io_pack.restype = ctypes.c_int32
        self._library.d2s_coreml_io_pack_rgb.argtypes = pack_argtypes
        self._library.d2s_coreml_io_pack_rgb.restype = ctypes.c_int32
        self._library.d2s_coreml_io_release.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self._library.d2s_coreml_io_release.restype = ctypes.c_int32
        self._library.d2s_coreml_io_last_error.argtypes = [ctypes.c_void_p]
        self._library.d2s_coreml_io_last_error.restype = ctypes.c_char_p
        self._library.d2s_coreml_io_destroy.argtypes = [ctypes.c_void_p]
        self._library.d2s_coreml_io_destroy.restype = None

        error = ctypes.create_string_buffer(512)
        compute_units = {"cpu": 0, "cpu_and_gpu": 1, "ane": 3}.get(
            os.environ.get("D2S_COREML_COMPUTE_UNITS", "all").strip().lower(), 2
        )
        self._handle = self._library.d2s_coreml_io_create(
            str(model_path).encode(),
            int(input_width),
            int(input_height),
            compute_units,
            error,
            len(error),
        )
        if not self._handle:
            raise RuntimeError(error.value.decode(errors="replace") or "native CoreML bridge creation failed")
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.last_error = ""
        # Keep native CoreML output on the same fast FXAA path as Vulkan and
        # CUDA/ROCm. Inference and its synchronization remain untouched.
        self.smaa_enabled = False

    @classmethod
    def try_create(cls, model_path: str | Path, input_width: int, input_height: int):
        try:
            return cls(model_path, input_width, input_height), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _ensure_lifetime_state(self):
        condition = getattr(self, "_state_condition", None)
        if condition is None:
            condition = threading.Condition()
            self._state_condition = condition
            self._active_calls = 0
            self._outstanding_frames = 0
            self._closing = False
        return condition

    def _begin_call(self, *, allow_closing: bool = False):
        condition = self._ensure_lifetime_state()
        with condition:
            handle = getattr(self, "_handle", None)
            if not handle or (self._closing and not allow_closing):
                raise RuntimeError("native CoreML bridge is closed")
            self._active_calls += 1
            return handle

    def _end_call(self) -> None:
        condition = self._ensure_lifetime_state()
        with condition:
            self._active_calls = max(0, self._active_calls - 1)
            condition.notify_all()

    def _destroy_if_ready(self) -> None:
        condition = self._ensure_lifetime_state()
        with condition:
            if (
                not self._closing
                or self._active_calls
                or self._outstanding_frames
            ):
                return
            handle, self._handle = getattr(self, "_handle", None), None
        if handle:
            self._library.d2s_coreml_io_destroy(handle)

    def predict(self, sck_frame: Any, frame_id: int) -> NativeCoreMLFrame:
        pixel_buffer = getattr(sck_frame, "pixel_buffer", None) or sck_frame
        try:
            import objc

            pointer = ctypes.c_void_p(int(objc.pyobjc_id(pixel_buffer)))
        except Exception as exc:
            raise RuntimeError(f"cannot obtain CVPixelBuffer pointer: {exc}") from exc
        result = _NativeResult()
        handle = self._begin_call()
        native_frame = None
        prediction_succeeded = False
        try:
            code = self._library.d2s_coreml_io_predict(
                handle, pointer, int(frame_id), ctypes.byref(result)
            )
            if code != 0:
                self._update_error(handle)
                if code == -2:
                    raise NativeCoreMLBusy(
                        self.last_error or "native CoreML resource ring is full"
                    )
                raise RuntimeError(
                    self.last_error or f"native CoreML prediction failed ({code})"
                )
            prediction_succeeded = True
            native_frame = NativeCoreMLFrame(
                bridge=self,
                slot=int(result.slot),
                frame_id=int(frame_id),
                source_width=int(result.source_width),
                source_height=int(result.source_height),
                depth_width=int(result.depth_width),
                depth_height=int(result.depth_height),
                finite_depth=bool(result.finite_depth),
                nonfinite_count=int(result.nonfinite_count),
                preprocess_ms=float(result.preprocess_ms),
                model_ms=float(result.model_ms),
                postprocess_ms=float(result.postprocess_ms),
                input_shared=bool(result.input_shared),
                output_backing_used=bool(result.output_backing_used),
                output_zero_copy=bool(result.output_zero_copy),
                normalize_lo=float(result.normalize_lo),
                normalize_hi=float(result.normalize_hi),
                raw_normalize_lo=float(result.raw_normalize_lo),
                raw_normalize_hi=float(result.raw_normalize_hi),
                normalization_history_reset=bool(result.normalization_history_reset),
            )
            condition = self._ensure_lifetime_state()
            with condition:
                self._outstanding_frames += 1
            return native_frame
        except Exception:
            if prediction_succeeded and native_frame is None and int(result.slot) >= 0:
                try:
                    self._library.d2s_coreml_io_release(handle, int(result.slot))
                except Exception:
                    pass
            raise
        finally:
            self._end_call()

    def pack(
        self,
        frame: NativeCoreMLFrame,
        destination: ctypes.c_void_p,
        destination_size: int,
        output_width: int,
        output_height: int,
        output_format: str,
        *,
        rgb: bool = False,
    ) -> None:
        try:
            eye_offset = float(os.environ.get("D2S_METAL_WARP_IPD", "0.064")) / 2.0
            depth_strength = 0.1 * float(
                os.environ.get("D2S_METAL_WARP_DEPTH_STRENGTH", "4.0")
            )
            convergence = float(os.environ.get("D2S_METAL_WARP_CONVERGENCE", "0.0"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid native warp configuration: {exc}") from exc
        values = dict(getattr(frame, "warp_config", None) or {})
        config = _NativeWarpConfig(
            float(values.get("depth_strength", depth_strength)),
            float(values.get("max_disparity_px", 48.0)),
            float(values.get("convergence", convergence)),
            float(values.get("edge_threshold", 0.04)),
            float(values.get("fill_strength", 0.0)),
            int(values.get("fill_radius", 0)),
            int(values.get("mask_feather_radius", 0)),
            int(values.get("symmetric", 1)),
            int(values.get("layers", 2)),
            float(values.get("softness", 0.08)),
            float(values.get("foreground_scale", 1.0)),
            float(values.get("midground_scale", 1.0)),
            float(values.get("background_scale", 1.0)),
            int(values.get("edge_dilation", 2)),
            int(values.get("screen_edge_suppression", 0)),
            int(values.get("hole_fill_mode", 2)),
            int(values.get("occlusion_enabled", 1)),
            float(values.get("depth_pop", 0.0)),
            float(values.get("antialias_strength", 0.0)),
            native_anaglyph_method_id(values.get("anaglyph_method", "red_cyan")),
            int(
                values.get(
                    "edge_aa_enabled",
                    output_edge_aa_enabled(),
                )
            ),
        )
        handle = self._begin_call(allow_closing=True)
        try:
            pack_function = (
                self._library.d2s_coreml_io_pack_rgb
                if rgb
                else self._library.d2s_coreml_io_pack
            )
            code = pack_function(
                handle,
                int(frame.slot),
                destination,
                int(destination_size),
                int(output_width),
                int(output_height),
                native_output_format_id(output_format),
                ctypes.byref(config),
                eye_offset,
                depth_strength,
                convergence,
                1.5,
            )
            if code != 0:
                self._update_error(handle)
                raise RuntimeError(
                    self.last_error or f"native Metal pack failed ({code})"
                )
        finally:
            self._end_call()

    def release(self, slot: int) -> None:
        condition = self._ensure_lifetime_state()
        with condition:
            handle = getattr(self, "_handle", None)
            if handle:
                self._active_calls += 1
        try:
            if handle:
                self._library.d2s_coreml_io_release(handle, int(slot))
        finally:
            if handle:
                self._end_call()
            with condition:
                self._outstanding_frames = max(0, self._outstanding_frames - 1)
                condition.notify_all()
            self._destroy_if_ready()

    def _update_error(self, handle=None) -> None:
        value = self._library.d2s_coreml_io_last_error(
            self._handle if handle is None else handle
        )
        self.last_error = value.decode(errors="replace") if value else ""

    def close(self) -> None:
        condition = self._ensure_lifetime_state()
        with condition:
            self._closing = True
            while self._active_calls:
                condition.wait()
        self._destroy_if_ready()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "NativeCoreMLBusy",
    "NativeCoreMLFrame",
    "NativeCoreMLIOBridge",
    "native_io_enabled",
]
