from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, TypeAlias


OutputResolution: TypeAlias = int | tuple[int, int]


@dataclass(frozen=True)
class CaptureConfig:
    output_resolution: OutputResolution = 1080
    fps: int = 60
    window_title: str | None = None
    capture_mode: str = "Monitor"
    monitor_index: int = 1
    capture_tool: str | None = None
    os_name: str | None = None
    fps_provider: Callable[[], int] | None = None


class FrameCopyMode(Enum):
    NONE = "none"
    CLONE = "clone"
    COPY = "copy"
    CPU_NUMPY = "cpu_numpy"
    GPU_TENSOR = "gpu_tensor"


@dataclass(frozen=True)
class CapturedFrame:
    frame: Any
    target_height: OutputResolution
    timestamp: float
    capture_tool: str = ""
    capture_mode: str = ""
    monitor_index: int = 0
    window_title: str = ""
    capture_size: tuple[int, int] | None = None
    frame_raw_type: str = ""
    frame_raw_device: str = ""
    frame_raw_dtype: str = ""
    copy_mode: FrameCopyMode = FrameCopyMode.COPY
    original_format: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional borrowed producer resource retained for a GPU-native consumer.
    native_resource: Any | None = None
    # Optional CPU compatibility frame kept separate from the native output.
    cpu_compat_frame: Any | None = None
    # Owned zero-copy capture frame (macOS ScreenCaptureKit): a
    # CVPixelBuffer+CVMetalTexture pair handed capture -> runtime -> viewer
    # along a single-owner chain; GC frees it on drop.
    sck_zero_copy: Any | None = None


def _frame_raw_type(frame: Any) -> str:
    frame_type = type(frame)
    module = getattr(frame_type, "__module__", "")
    name = getattr(frame_type, "__qualname__", getattr(frame_type, "__name__", ""))
    return f"{module}.{name}" if module and module != "builtins" else str(name)


def _frame_raw_device(frame: Any) -> str:
    device = getattr(frame, "device", "")
    if callable(device):
        try:
            device = device()
        except Exception:
            device = ""
    return str(device) if device is not None else ""


def _frame_raw_dtype(frame: Any) -> str:
    dtype = getattr(frame, "dtype", "")
    return str(dtype) if dtype is not None else ""


def _capture_size(frame: Any) -> tuple[int, int] | None:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        return int(shape[3]), int(shape[2])
    if len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    width = getattr(frame, "width", None)
    height = getattr(frame, "height", None)
    if width is not None and height is not None:
        return int(width), int(height)
    return None


def capture_frame_from_raw(
    frame: Any,
    target_height: OutputResolution,
    timestamp: float,
    *,
    config: CaptureConfig | None = None,
    copy_mode: FrameCopyMode = FrameCopyMode.COPY,
    original_format: str = "",
    metadata: dict[str, Any] | None = None,
    capture_size: tuple[int, int] | None = None,
    frame_raw_type: str | None = None,
    frame_raw_device: str | None = None,
    frame_raw_dtype: str | None = None,
    native_resource: Any | None = None,
    cpu_compat_frame: Any | None = None,
    sck_zero_copy: Any | None = None,
) -> CapturedFrame:
    return CapturedFrame(
        frame=frame,
        target_height=target_height,
        timestamp=timestamp,
        capture_tool=str(config.capture_tool or "") if config is not None else "",
        capture_mode=str(config.capture_mode or "") if config is not None else "",
        monitor_index=int(config.monitor_index) if config is not None else 0,
        window_title=str(config.window_title or "") if config is not None else "",
        capture_size=capture_size if capture_size is not None else _capture_size(frame),
        frame_raw_type=frame_raw_type if frame_raw_type is not None else _frame_raw_type(frame),
        frame_raw_device=frame_raw_device if frame_raw_device is not None else _frame_raw_device(frame),
        frame_raw_dtype=frame_raw_dtype if frame_raw_dtype is not None else _frame_raw_dtype(frame),
        copy_mode=copy_mode,
        original_format=original_format,
        metadata=dict(metadata or {}),
        native_resource=native_resource,
        cpu_compat_frame=cpu_compat_frame,
        sck_zero_copy=sck_zero_copy,
    )


def native_resource_contract(resource: Any) -> dict[str, Any]:
    """Return conservative metadata for a borrowed producer resource."""
    adapter_luid = int(getattr(resource, "adapter_luid", 0) or 0)
    adapter_uuid = getattr(resource, "adapter_uuid", None) or getattr(resource, "device_uuid", None)
    pci_bdf = getattr(resource, "pci_bdf", None) or getattr(resource, "drm_pci_bdf", None)
    width = getattr(resource, "width", None)
    height = getattr(resource, "height", None)
    adapter_identity = (
        f"luid:{adapter_luid:016x}"
        if adapter_luid
        else f"uuid:{adapter_uuid}"
        if adapter_uuid
        else f"pci:{pci_bdf}"
        if pci_bdf
        else None
    )
    return {
        "resource_kind": str(
            getattr(resource, "resource_kind", "")
            or getattr(resource, "kind", "")
            or type(resource).__name__
        ),
        "resource_format": str(getattr(resource, "format", "") or "unknown"),
        "resource_width": int(width) if width is not None else None,
        "resource_height": int(height) if height is not None else None,
        "adapter_luid": adapter_luid,
        "adapter_uuid": str(adapter_uuid) if adapter_uuid else None,
        "pci_bdf": str(pci_bdf) if pci_bdf else None,
        "adapter_identity": adapter_identity,
        "resource_lifecycle": "borrowed_until_frame_release",
    }


def compatibility_frame(item: Any) -> Any:
    """Return the explicit CPU compatibility frame, if one was retained."""
    if isinstance(item, CapturedFrame):
        if item.cpu_compat_frame is not None:
            return item.cpu_compat_frame
        return item.frame
    return item


def release_native_resource(item: Any) -> bool:
    """Release only an explicit borrowed-resource lease, never arbitrary objects."""
    resource = getattr(item, "native_resource", None)
    release = getattr(resource, "release", None)
    if not callable(release):
        return False
    try:
        release()
        return True
    except Exception:
        return False


def capture_frame_from_native_texture(
    resource: Any,
    target_height: OutputResolution,
    timestamp: float,
    *,
    config: CaptureConfig | None = None,
) -> CapturedFrame:
    """Create a borrowed native-texture frame without mapping it to the CPU."""
    metadata = {
        "backend": "desktop_duplication",
        "capture_gpu": True,
        "gpu_to_cpu": False,
        "gpu_copy_count": 0,
        # The encoder/inference consumer has not yet been verified.
        "zero_copy": False,
        "zero_copy_ready": False,
        **native_resource_contract(resource),
    }
    return capture_frame_from_raw(
        resource,
        target_height,
        timestamp,
        config=config,
        copy_mode=FrameCopyMode.NONE,
        original_format=str(getattr(resource, "format", "BGRA8")),
        frame_raw_device="d3d11",
        native_resource=resource,
        metadata=metadata,
        capture_size=(int(resource.width), int(resource.height)),
    )


def ensure_captured_frame(
    item: CapturedFrame | tuple[Any, OutputResolution, float],
    *,
    config: CaptureConfig | None = None,
) -> CapturedFrame:
    if isinstance(item, CapturedFrame):
        return item
    frame, target_height, timestamp = item
    return capture_frame_from_raw(frame, target_height, timestamp, config=config)


class CaptureSource(Protocol):
    def grab(self): ...
    def stop(self) -> None: ...


FrameCallback = Callable[[CapturedFrame], None]
ErrorCallback = Callable[[BaseException], None]
StateCallback = Callable[[Any | None, Any | None], None]
Predicate = Callable[[], bool]
PausedCallback = Callable[[str], None]


class CaptureRunner(Protocol):
    def run(
        self,
        *,
        shutdown_event: Any,
        on_frame: FrameCallback,
        on_error: ErrorCallback | None = None,
        on_closed: Callable[[], None] | None = None,
        is_paused: Predicate | None = None,
        is_hard_idle: Predicate | None = None,
        on_paused: PausedCallback | None = None,
        on_session_update: StateCallback | None = None,
        on_tick: Callable[[], None] | None = None,
    ) -> None: ...
