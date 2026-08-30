from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from threading import RLock
from typing import Any

try:
    import triton as _triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on installations without Triton
    _triton = None
    tl = None


def _ensure_windows_cc() -> None:
    """Point Triton's C compiler at its bundled TinyCC when MSVC is absent.

    On Windows, ``triton.runtime.build.get_cc()`` prefers the ROCm SDK
    clang-cl, which needs MSVC/Windows-SDK headers for ``stdlib.h``.  On
    machines without Visual Studio Build Tools every ``hip_utils`` compile then
    fails and every Triton kernel launch raises.  Triton ships its own TinyCC
    (``triton/runtime/tcc/tcc.exe``) with a bundled libc for exactly this case;
    selecting it keeps the ROCm Triton backend usable without MSVC.
    """
    if os.name != "nt" or os.environ.get("CC"):
        return
    try:
        import sysconfig

        from triton.windows_utils import find_msvc

        msvc_bin, _, _ = find_msvc(env_only=False)
        if msvc_bin:
            return  # real MSVC present; keep Triton's default compiler
        tcc = os.path.join(
            sysconfig.get_paths()["platlib"],
            "triton",
            "runtime",
            "tcc",
            "tcc.exe",
        )
        if os.path.exists(tcc):
            os.environ["CC"] = tcc
    except Exception:  # pragma: no cover - best-effort environment fix
        pass


@dataclass(frozen=True, slots=True)
class TritonRuntimeInfo:
    """Result of a real Triton device probe."""

    available: bool
    vendor: str
    backend: str
    reason: str
    device: str
    triton_version: str
    torch_version: str
    is_windows: bool


_LOCK = RLock()
_CACHE: dict[str, TritonRuntimeInfo] = {}


def _device_key(device: Any) -> str:
    value = str(device)
    if value == "cuda":
        return "cuda:0"
    return value


def _unavailable(
    *,
    reason: str,
    device: Any = "unknown",
    triton_version: str = "unknown",
    torch_version: str = "unknown",
) -> TritonRuntimeInfo:
    return TritonRuntimeInfo(
        available=False,
        vendor="unknown",
        backend="unavailable",
        reason=str(reason),
        device=str(device),
        triton_version=str(triton_version),
        torch_version=str(torch_version),
        is_windows=platform.system().lower() == "windows",
    )


if _triton is not None:

    @_triton.jit
    def _probe_kernel(x, out, n: tl.constexpr):
        offsets = tl.program_id(0) * 32 + tl.arange(0, 32)
        mask = offsets < n
        values = tl.load(x + offsets, mask=mask, other=0.0)
        tl.store(out + offsets, values + 1.0, mask=mask)
else:
    _probe_kernel = None


def probe_triton_runtime(device: Any = None, *, force: bool = False) -> TritonRuntimeInfo:
    """Import and execute a tiny Triton kernel on the requested torch device.

    PyTorch ROCm exposes AMD tensors through the ``cuda`` device API. The probe
    therefore tests the actual compiler/runtime instead of inferring support
    from a vendor ID or package name.
    """

    _ensure_windows_cc()
    try:
        import torch
    except Exception as exc:
        return _unavailable(reason=f"torch_import_failed:{type(exc).__name__}")

    if device is None:
        try:
            device = torch.device("cuda", torch.cuda.current_device())
        except Exception as exc:
            return _unavailable(reason=f"cuda_device_unavailable:{type(exc).__name__}")
    else:
        device = torch.device(device)
    key = _device_key(device)
    if not force:
        with _LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return cached

    torch_version = str(getattr(torch, "__version__", "unknown"))
    try:
        import triton
    except Exception as exc:
        result = _unavailable(
            reason=f"triton_import_failed:{type(exc).__name__}",
            device=device,
            torch_version=torch_version,
        )
        with _LOCK:
            _CACHE[key] = result
        return result

    triton_version = str(getattr(triton, "__version__", "unknown"))
    if device.type != "cuda":
        result = _unavailable(
            reason=f"unsupported_torch_device:{device.type}",
            device=device,
            triton_version=triton_version,
            torch_version=torch_version,
        )
        with _LOCK:
            _CACHE[key] = result
        return result
    vendor = _detect_gpu_vendor(torch, device)
    if vendor not in {"nvidia", "amd"}:
        result = TritonRuntimeInfo(
            available=False,
            vendor=vendor,
            backend="unsupported_vendor",
            reason=f"unsupported_gpu_vendor:{vendor}",
            device=str(device),
            triton_version=triton_version,
            torch_version=torch_version,
            is_windows=platform.system().lower() == "windows",
        )
        with _LOCK:
            _CACHE[key] = result
        return result
    try:
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("torch.cuda.is_available() is false")
        if _probe_kernel is None:
            raise RuntimeError("Triton probe kernel is unavailable")
        size = 32
        source = torch.zeros(size, device=device, dtype=torch.float32)
        target = torch.empty_like(source)
        _probe_kernel[(1,)](source, target, size)
        torch.cuda.synchronize(device)
        if not bool(torch.allclose(target, torch.ones_like(target))):
            raise RuntimeError("probe output mismatch")
    except Exception as exc:
        result = _unavailable(
            reason=f"triton_kernel_probe_failed:{type(exc).__name__}:{exc}",
            device=device,
            triton_version=triton_version,
            torch_version=torch_version,
        )
        with _LOCK:
            _CACHE[key] = result
        return result

    backend = "amd_rocm" if vendor == "amd" else "nvidia_cuda"
    result = TritonRuntimeInfo(
        available=True,
        vendor=vendor,
        backend=backend,
        reason="ok",
        device=str(device),
        triton_version=triton_version,
        torch_version=torch_version,
        is_windows=platform.system().lower() == "windows",
    )
    with _LOCK:
        _CACHE[key] = result
    return result


def triton_runtime_available(device: Any) -> bool:
    """Return whether a real Triton kernel is usable on ``device``."""

    info = probe_triton_runtime(device)
    return bool(info.available and info.vendor in {"nvidia", "amd"})


def _detect_gpu_vendor(torch_module: Any, device: Any) -> str:
    """Detect only the vendors covered by the Triton policy."""

    if getattr(getattr(torch_module, "version", None), "hip", None):
        return "amd"
    try:
        name = str(torch_module.cuda.get_device_properties(device).name).upper()
    except Exception:
        return "unknown"
    if any(token in name for token in ("NVIDIA", "GEFORCE", "RTX", "GTX", "QUADRO", "TESLA", "TITAN")):
        return "nvidia"
    if any(token in name for token in ("AMD", "RADEON", "INSTINCT", "FIREPRO")):
        return "amd"
    return "unknown"


def clear_triton_probe_cache() -> None:
    """Clear cached probe results for tests and runtime reinitialization."""

    with _LOCK:
        _CACHE.clear()
