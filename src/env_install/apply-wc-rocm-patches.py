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
_ROCM_WINDOW_EXCLUSION_HELPER = """# D2S: global capture fallback.


def _d2s_set_window_capture_affinity(hwnd, affinity):
    try:
        setter = ctypes.windll.user32.SetWindowDisplayAffinity
        setter.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        setter.restype = ctypes.c_int
        return bool(setter(ctypes.c_void_p(int(hwnd)), int(affinity)))
    except Exception:
        return False
"""
_OLD_ROCM_ADAPTER_LUID_MARKER = "# D2S: map HIP ordinals to AMD DXGI adapter identities."
_ROCM_ADAPTER_LUID_MARKER = "# D2S: use HIP's per-device graphics LUID (works with multiple AMD GPUs)."


def patched_source(source: str) -> str:
    newline = "\r\n" if "\r\n" in source else "\n"
    has_capture_binding = "class WindowsCapture:" in source and "_NativeWcCapture" in source
    changes = (
        (
            (
                f"class Frame:{newline}"
                f"    def __init__(self, tensor: torch.Tensor, width: int, height: int):{newline}"
                f"        self.frame_buffer = tensor  # uint8 BGRA tensor on ROCm device{newline}"
                f"        self.width        = width{newline}"
                f"        self.height       = height{newline}"
            ),
            (
                f"class Frame:{newline}"
                f"    def __init__(self, tensor: torch.Tensor, width: int, height: int, configuration_iteration: int = 0):{newline}"
                f"        self.frame_buffer = tensor  # uint8 BGRA tensor on ROCm device{newline}"
                f"        self.width        = width{newline}"
                f"        self.height       = height{newline}"
                f"        self.configuration_iteration = int(configuration_iteration){newline}"
            ),
        ),
        (
            (
                f"        use_gpu_fence: bool = True,{newline}"
                f"    ):{newline}"
            ),
            (
                f"        use_gpu_fence: bool = True,{newline}"
                f"        excluded_window_handles: Optional[list[int]] = None,{newline}"
                f"    ):{newline}"
            ),
        ),
        (
            f"        self._window_hwnd = window_hwnd{newline}",
            f"        self._window_hwnd = window_hwnd{newline}"
            f"        self._native_window_exclusion_available = callable(getattr(_NativeWcCapture, 'set_excluded_windows', None)){newline}"
            f"        self._excluded_window_handles = tuple(sorted({{int(hwnd) for hwnd in (excluded_window_handles or ())}})){newline}"
            f"        if any(hwnd <= 0 for hwnd in self._excluded_window_handles):{newline}"
            f"            raise ValueError('Window handles must be positive'){newline}"
            f"        self._window_exclusion_requested = excluded_window_handles is not None{newline}"
            f"        self._required_configuration_iteration = 0{newline}",
        ),
        (
            (
                f"            use_gpu_fence=use_gpu_fence,{newline}"
                f"        ){newline}"
            ),
            (
                f"            use_gpu_fence=use_gpu_fence,{newline}"
                f"            excluded_window_handles=({newline}"
                f"                list(self._excluded_window_handles){newline}"
                f"                if self._window_exclusion_requested and self._native_window_exclusion_available{newline}"
                f"                else None{newline}"
                f"            ),{newline}"
                f"        ){newline}"
            ),
        ),
        (
            f"    def _make_inner(self, use_gpu_fence: bool):{newline}",
            f"    @property{newline}"
            f"    def window_exclusion_available(self) -> bool:{newline}"
            f"        return self._native_window_exclusion_available{newline}"
            f"{newline}"
            f"    @property{newline}"
            f"    def window_exclusion_configuration_iteration(self) -> int:{newline}"
            f"        native_value = getattr(self._inner, 'required_configuration_iteration', 0){newline}"
            f"        native_iteration = int(native_value() if callable(native_value) else native_value) or 0{newline}"
            f"        if native_iteration > self._required_configuration_iteration:{newline}"
            f"            self._required_configuration_iteration = native_iteration{newline}"
            f"        return self._required_configuration_iteration{newline}"
            f"{newline}"
            f"    def set_excluded_windows(self, window_handles) -> int:{newline}"
            f"        if not self._native_window_exclusion_available:{newline}"
            f"            raise RuntimeError('Native WGC window exclusion is unavailable'){newline}"
            f"        handles = tuple(sorted({{int(hwnd) for hwnd in window_handles}})){newline}"
            f"        if any(hwnd <= 0 for hwnd in handles):{newline}"
            f"            raise ValueError('Window handles must be positive'){newline}"
            f"        iteration = int(self._inner.set_excluded_windows(list(handles))){newline}"
            f"        self._excluded_window_handles = handles{newline}"
            f"        self._window_exclusion_requested = True{newline}"
            f"        self._required_configuration_iteration = iteration{newline}"
            f"        return iteration{newline}"
            f"{newline}"
            f"    def _make_inner(self, use_gpu_fence: bool):{newline}"
        ),
        (
            f"                buf_idx = gpu_frame.buf_idx  # 0 or 1, set by Rust ping-pong{newline}",
            f"                buf_idx = gpu_frame.buf_idx  # 0 or 1, set by Rust ping-pong{newline}"
            f"                native_value = getattr(self._inner, 'required_configuration_iteration', 0){newline}"
            f"                native_iteration = int(native_value() if callable(native_value) else native_value) or 0{newline}"
            f"                if native_iteration > self._required_configuration_iteration:{newline}"
            f"                    self._required_configuration_iteration = native_iteration{newline}"
            f"                if int(getattr(gpu_frame, 'configuration_iteration', 0)) < self._required_configuration_iteration:{newline}"
            f"                    continue{newline}",
        ),
        (
            f"                self.frame_handler(Frame(tensor[:oh, :ow], ow, oh), self._control){newline}",
            f"                self.frame_handler(Frame({newline}"
            f"                    tensor[:oh, :ow], ow, oh,{newline}"
            f"                    int(getattr(gpu_frame, 'configuration_iteration', 0)),{newline}"
            f"                ), self._control){newline}",
        ),
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
        (
            f"    def _make_inner(self, use_gpu_fence: bool):{newline}",
            f"    def _d2s_apply_global_window_exclusion(self):{newline}"
            f"        if not self._window_exclusion_requested:{newline}"
            f"            return{newline}"
            f"        failed = [hwnd for hwnd in self._excluded_window_handles{newline}"
            f"                  if not _d2s_set_window_capture_affinity(hwnd, 0x11)]{newline}"
            f"        print('[wc-rocm] Falling back to global capture exclusion; remote SBS will be hidden', flush=True){newline}"
            f"        if failed:{newline}"
            f"            print(f'[wc-rocm] Could not exclude HWNDs {{failed}} from capture', flush=True){newline}"
            f"{newline}"
            f"    def _make_inner(self, use_gpu_fence: bool):{newline}"
        ),
        (
            f"        self._inner.start(){newline}",
            f"        if self._window_exclusion_requested and not self._native_window_exclusion_available:{newline}"
            f"            self._d2s_apply_global_window_exclusion(){newline}"
            f"        try:{newline}"
            f"            self._inner.start(){newline}"
            f"        except Exception:{newline}"
            f"            self._d2s_apply_global_window_exclusion(){newline}"
            f"            raise{newline}",
        ),
        (
            f"            err = self._inner.get_last_error(){newline}"
            f"            if err:{newline}"
            f"                raise RuntimeError(err){newline}",
            f"            err = self._inner.get_last_error(){newline}"
            f"            if err:{newline}"
            f"                self._d2s_apply_global_window_exclusion(){newline}"
            f"                raise RuntimeError(err){newline}",
        ),
        (
            f"                        err = self._inner.get_last_error(){newline}"
            f"                        if err:{newline}"
            f"                            raise RuntimeError(err){newline}",
            f"                        err = self._inner.get_last_error(){newline}"
            f"                        if err:{newline}"
            f"                            self._d2s_apply_global_window_exclusion(){newline}"
            f"                            raise RuntimeError(err){newline}",
        ),
    )
    for old, new in changes:
        if not has_capture_binding and any(
            marker in old + new
            for marker in (
                "excluded_window_handles",
                "window_exclusion",
                "configuration_iteration",
                "set_excluded_windows",
            )
        ):
            continue
        if (
            "def set_excluded_windows(self, window_handles)" in source
            and old.strip() == "def _make_inner(self, use_gpu_fence: bool):"
            and "def window_exclusion_available" in new
        ):
            continue
        if new in source:
            continue
        if source.count(old) != 1:
            raise ValueError(
                "Unsupported wc-rocm source layout; no file was changed. "
                f"Missing unique source block {old!r}"
            )
        source = source.replace(old, new, 1)
    if has_capture_binding:
        property_anchor = f"    def set_excluded_windows(self, window_handles) -> int:{newline}"
        property_source = (
            f"    @property{newline}"
            f"    def window_exclusion_configuration_iteration(self) -> int:{newline}"
            f"        native_value = getattr(self._inner, 'required_configuration_iteration', 0){newline}"
            f"        native_iteration = int(native_value() if callable(native_value) else native_value) or 0{newline}"
            f"        if native_iteration > self._required_configuration_iteration:{newline}"
            f"            self._required_configuration_iteration = native_iteration{newline}"
            f"        return self._required_configuration_iteration{newline}"
            f"{newline}"
        )
        if "def window_exclusion_configuration_iteration" not in source:
            if source.count(property_anchor) != 1:
                raise ValueError(
                    "Unsupported wc-rocm source layout; no file was changed. "
                    "Missing unique window-exclusion property anchor"
                )
            source = source.replace(
                property_anchor,
                property_source + property_anchor,
                1,
            )
        iteration_anchor = (
            f"                if int(getattr(gpu_frame, 'configuration_iteration', 0)) "
            f"< self._required_configuration_iteration:{newline}"
        )
        iteration_source = (
            f"                native_value = getattr(self._inner, "
            f"'required_configuration_iteration', 0){newline}"
            f"                native_iteration = int(native_value() if callable(native_value) else native_value) or 0{newline}"
            f"                if native_iteration > self._required_configuration_iteration:{newline}"
            f"                    self._required_configuration_iteration = native_iteration{newline}"
        )
        if "native_iteration = int(getattr(self._inner" not in source:
            if source.count(iteration_anchor) != 1:
                raise ValueError(
                    "Unsupported wc-rocm source layout; no file was changed. "
                    "Missing unique configuration-iteration gate"
                )
            source = source.replace(
                iteration_anchor,
                iteration_source + iteration_anchor,
                1,
            )
    old_patch_offset = source.rfind(_OLD_ROCM_ADAPTER_LUID_MARKER)
    if old_patch_offset >= 0:
        source = source[:old_patch_offset]
    if _ROCM_ADAPTER_LUID_MARKER not in source:
        if not source.endswith(("\n", "\r")):
            source += newline
        source += newline + _ROCM_ADAPTER_LUID_PATCH.replace("\n", newline) + newline
    if "# D2S: global capture fallback." not in source:
        if not source.endswith(("\n", "\r")):
            source += newline
        source += newline + _ROCM_WINDOW_EXCLUSION_HELPER.replace("\n", newline) + newline
    compile(source, "wc_rocm/__init__.py", "exec")
    return source


def main() -> int:
    if sys.platform != "win32":
        return 0
    from wc_capture_build import build_native_capture

    runtime = Path(sys.executable).resolve().parent
    bundled_root = Path(__file__).resolve().parents[1]
    if runtime.parent != bundled_root or runtime.name != "python3":
        raise RuntimeError("Run this patch using the bundled src/python3 runtime")

    target = (runtime / "Lib/site-packages/wc_rocm/__init__.py").resolve()
    target.relative_to(runtime)
    original = target.read_bytes()
    source = original.decode("utf-8")
    patched = patched_source(source)
    native_capture_built = build_native_capture(
        runtime,
        crate_name="wc_rocm",
        package_name="wc_rocm",
        module_name="_wc_rocm",
    )
    if patched != source:
        target.write_bytes(patched.encode("utf-8"))
    print(
        "[wc-rocm] bundled HIP capture ready; "
        f"WGC window exclusions {'enabled' if native_capture_built else 'unavailable'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
