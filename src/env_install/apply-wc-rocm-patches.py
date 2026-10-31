"""Use HIP adapter LUIDs for multi-GPU capture and pace bundled transfers.

Apply this patch only to the ROCm Python runtime bundled with Desktop2Stereo.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROCM_ADAPTER_LUID_PATCH = """# D2S: use HIP's per-device graphics LUID (works with multiple AMD GPUs).
def _d2s_get_luid(device_id: int = 0):
    try:
        luid = ctypes.create_string_buffer(8)
        node_mask = ctypes.c_uint()
        hip = _get_hip()
        get_device_luid = hip.hipDeviceGetLuid
        get_device_luid.argtypes = [
            ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint), ctypes.c_int,
        ]
        get_device_luid.restype = ctypes.c_int
        if get_device_luid(luid, ctypes.byref(node_mask), int(device_id)) != 0:
            return (0, 0)
        packed = int.from_bytes(luid.raw, \"little\")
        low = packed & 0xFFFFFFFF
        high = ctypes.c_int32(packed >> 32).value
        return (low, high)
    except Exception:
        return (0, 0)


get_luid = _d2s_get_luid
"""
_OLD_ROCM_ADAPTER_LUID_MARKER = "# D2S: map HIP ordinals to AMD DXGI adapter identities."
_ROCM_ADAPTER_LUID_MARKER = "# D2S: use HIP's per-device graphics LUID (works with multiple AMD GPUs)."


def patched_source(source: str) -> str:
    newline = "\r\n" if "\r\n" in source else "\n"
    changes = (
        (
            f"        self._stream        = None{newline}",
            (
                f"        self._stream        = None{newline}"
                f"        self._minimum_update_interval_s = max(0.0, float(minimum_update_interval or 0) / 1000.0){newline}"
            ),
        ),
        (
            (
                f"            while self._running:{newline}"
                f"                res = self._inner.get_frame(self._last_id, timeout=0.1){newline}"
            ),
            (
                f"            next_transfer_time = 0.0{newline}"
                f"            while self._running:{newline}"
                f"                # Pace before fetching and importing the next D3D11 frame.{newline}"
                f"                if self._minimum_update_interval_s > 0.0:{newline}"
                f"                    delay = next_transfer_time - time.perf_counter(){newline}"
                f"                    if delay > 0.0:{newline}"
                f"                        time.sleep(delay){newline}"
                f"                    if not self._running:{newline}"
                f"                        break{newline}"
                f"                    next_transfer_time = time.perf_counter() + self._minimum_update_interval_s{newline}"
                f"                res = self._inner.get_frame(self._last_id, timeout=0.1){newline}"
            ),
        ),
    )
    for old, new in changes:
        if new in source:
            continue
        if source.count(old) != 1:
            raise ValueError("Unsupported wc-rocm source layout; no file was changed")
        source = source.replace(old, new, 1)
    old_patch_offset = source.rfind(_OLD_ROCM_ADAPTER_LUID_MARKER)
    if old_patch_offset >= 0:
        source = source[:old_patch_offset]
    if _ROCM_ADAPTER_LUID_MARKER not in source:
        if not source.endswith(("\n", "\r")):
            source += newline
        source += newline + _ROCM_ADAPTER_LUID_PATCH.replace("\n", newline) + newline
    compile(source, "wc_rocm/__init__.py", "exec")
    return source


def main() -> int:
    if sys.platform != "win32":
        return 0
    runtime = Path(sys.executable).resolve().parent
    bundled_root = Path(__file__).resolve().parents[1]
    if runtime.parent != bundled_root or runtime.name != "python3":
        raise RuntimeError("Run this patch using the bundled src/python3 runtime")

    target = (runtime / "Lib/site-packages/wc_rocm/__init__.py").resolve()
    target.relative_to(runtime)
    original = target.read_bytes()
    source = original.decode("utf-8")
    patched = patched_source(source)
    if patched != source:
        target.write_bytes(patched.encode("utf-8"))
    print("[wc-rocm] bundled HIP LUID mapping and capture pacing ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
