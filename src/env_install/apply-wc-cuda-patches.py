"""Honor wc-cuda's existing interval before D3D11/CUDA transfer.

The 0.1.3 Python wrapper accepts minimum_update_interval but ignores it.
Apply only to the Python runtime bundled beside this installer.
"""
from __future__ import annotations

import sys
from pathlib import Path

_WINDOW_EXCLUSION_HELPERS = """# D2S: global capture fallback.
def _d2s_set_window_capture_affinity(hwnd, affinity):
    try:
        setter = ctypes.windll.user32.SetWindowDisplayAffinity
        setter.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        setter.restype = ctypes.c_int
        return bool(setter(ctypes.c_void_p(int(hwnd)), int(affinity)))
    except Exception:
        return False
"""


def patched_source(source: str) -> str:
    has_capture_binding = "class WindowsCapture:" in source and "_NativeWcCapture" in source
    changes = (
        (
            "class Frame:\n"
            "    def __init__(self, tensor: torch.Tensor, width: int, height: int):\n"
            "        self.frame_buffer = tensor  # This is the PyTorch CUDA Tensor\n"
            "        self.width = width\n"
            "        self.height = height\n",
            "class Frame:\n"
            "    def __init__(self, tensor: torch.Tensor, width: int, height: int, configuration_iteration: int = 0):\n"
            "        self.frame_buffer = tensor  # This is the PyTorch CUDA Tensor\n"
            "        self.width = width\n"
            "        self.height = height\n"
            "        self.configuration_iteration = int(configuration_iteration)\n",
        ),
        (
            "        output_buffer_count: int = 3,\n"
            "    ):\n",
            "        output_buffer_count: int = 3,\n"
            "        excluded_window_handles: Optional[list[int]] = None,\n"
            "    ):\n",
        ),
        (
            "        self.device_id = device_id\n"
            "        self._inner = _NativeWcCapture(\n"
            "            luid=luid, monitor_index=monitor_index, window_hwnd=window_hwnd, window_title=window_name\n"
            "        )\n",
            "        self.device_id = device_id\n"
            "        self._native_window_exclusion_available = callable(getattr(_NativeWcCapture, 'set_excluded_windows', None))\n"
            "        self._excluded_window_handles = tuple(sorted({int(hwnd) for hwnd in (excluded_window_handles or ())}))\n"
            "        if any(hwnd <= 0 for hwnd in self._excluded_window_handles):\n"
            "            raise ValueError('Window handles must be positive')\n"
            "        self._window_exclusion_requested = excluded_window_handles is not None\n"
            "        self._required_configuration_iteration = 0\n"
            "        self._inner = _NativeWcCapture(\n"
            "            luid=luid, monitor_index=monitor_index, window_hwnd=window_hwnd, window_title=window_name,\n"
            "            cursor_capture=bool(cursor_capture),\n"
            "            excluded_window_handles=(\n"
            "                list(self._excluded_window_handles) if self._window_exclusion_requested and self._native_window_exclusion_available else None\n"
            "            ),\n"
            "        )\n",
        ),
        (
            "    def event(self, handler):\n",
            "    @property\n"
            "    def window_exclusion_available(self) -> bool:\n"
            "        return self._native_window_exclusion_available\n"
            "\n"
            "    @property\n"
            "    def window_exclusion_configuration_iteration(self) -> int:\n"
            "        native_value = getattr(self._inner, 'required_configuration_iteration', 0)\n"
            "        native_iteration = int(native_value() if callable(native_value) else native_value) or 0\n"
            "        if native_iteration > self._required_configuration_iteration:\n"
            "            self._required_configuration_iteration = native_iteration\n"
            "        return self._required_configuration_iteration\n"
            "\n"
            "    def set_excluded_windows(self, window_handles) -> int:\n"
            "        if not self._native_window_exclusion_available:\n"
            "            raise RuntimeError('Native WGC window exclusion is unavailable')\n"
            "        handles = tuple(sorted({int(hwnd) for hwnd in window_handles}))\n"
            "        if any(hwnd <= 0 for hwnd in handles):\n"
            "            raise ValueError('Window handles must be positive')\n"
            "        iteration = int(self._inner.set_excluded_windows(list(handles)))\n"
            "        self._excluded_window_handles = handles\n"
            "        self._window_exclusion_requested = True\n"
            "        self._required_configuration_iteration = iteration\n"
            "        return iteration\n"
            "\n"
            "    def _d2s_apply_global_window_exclusion(self):\n"
            "        if not self._window_exclusion_requested:\n"
            "            return\n"
            "        failed = [hwnd for hwnd in self._excluded_window_handles\n"
            "                  if not _d2s_set_window_capture_affinity(hwnd, 0x11)]\n"
            "        print('[wc-cuda] Falling back to global capture exclusion; remote SBS will be hidden', flush=True)\n"
            "        if failed:\n"
            "            print(f'[wc-cuda] Could not exclude HWNDs {failed} from capture', flush=True)\n"
            "\n"
            "    def event(self, handler):\n",
        ),
        (
            "                    ow, oh = gpu_frame.original_width, gpu_frame.original_height\n",
            "                    ow, oh = gpu_frame.original_width, gpu_frame.original_height\n"
            "                    native_value = getattr(self._inner, 'required_configuration_iteration', 0)\n"
            "                    native_iteration = int(native_value() if callable(native_value) else native_value) or 0\n"
            "                    if native_iteration > self._required_configuration_iteration:\n"
            "                        self._required_configuration_iteration = native_iteration\n"
            "                    if int(getattr(gpu_frame, 'configuration_iteration', 0)) < self._required_configuration_iteration:\n"
            "                        continue\n",
        ),
        (
            "                        frame = Frame(sliced_tensor, ow, oh)\n"
            "                        self.frame_handler(frame, self._control)\n",
            "                        frame = Frame(\n"
            "                            sliced_tensor, ow, oh,\n"
            "                            int(getattr(gpu_frame, 'configuration_iteration', 0)),\n"
            "                        )\n"
            "                        self.frame_handler(frame, self._control)\n",
        ),
        (
            "        self._inner.start()\n",
            "        if self._window_exclusion_requested and not self._native_window_exclusion_available:\n"
            "            self._d2s_apply_global_window_exclusion()\n"
            "        try:\n"
            "            self._inner.start()\n"
            "        except Exception:\n"
            "            self._d2s_apply_global_window_exclusion()\n"
            "            raise\n",
        ),
        (
            "            if err:\n"
            "                raise RuntimeError(err)\n",
            "            if err:\n"
            "                self._d2s_apply_global_window_exclusion()\n"
            "                raise RuntimeError(err)\n",
        ),
        (
            "                        if err:\n"
            "                            raise RuntimeError(err)\n",
            "                        if err:\n"
            "                            self._d2s_apply_global_window_exclusion()\n"
            "                            raise RuntimeError(err)\n",
        ),
        (
            "        self._stream = None\n",
            "        self._stream = None\n"
            "        self._minimum_update_interval_s = max(0.0, float(minimum_update_interval or 0) / 1000.0)\n",
        ),
        (
            "            while self._running:\n"
            "                # Use a small timeout to keep Python responsive (Ctrl+C)\n",
            "            next_transfer_time = 0.0\n"
            "            while self._running:\n"
            "                # Pace before interop, then fetch the newest native frame.\n"
            "                # Software pacing after the callback is too late to save GPU work.\n"
            "                if self._minimum_update_interval_s > 0.0:\n"
            "                    delay = next_transfer_time - time.perf_counter()\n"
            "                    if delay > 0.0:\n"
            "                        time.sleep(delay)\n"
            "                    if not self._running:\n"
            "                        break\n"
            "                    next_transfer_time = time.perf_counter() + self._minimum_update_interval_s\n"
            "                # Use a small timeout to keep Python responsive (Ctrl+C)\n",
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
            and old == "    def event(self, handler):\n"
        ):
            continue
        if new in source:
            continue
        if source.count(old) != 1:
            raise ValueError(
                "Unsupported wc-cuda source layout; no file was changed. "
                f"Missing unique source block {old!r}"
            )
        source = source.replace(old, new, 1)
    if has_capture_binding:
        property_anchor = "    def set_excluded_windows(self, window_handles) -> int:\n"
        property_source = (
            "    @property\n"
            "    def window_exclusion_configuration_iteration(self) -> int:\n"
            "        native_value = getattr(self._inner, 'required_configuration_iteration', 0)\n"
            "        native_iteration = int(native_value() if callable(native_value) else native_value) or 0\n"
            "        if native_iteration > self._required_configuration_iteration:\n"
            "            self._required_configuration_iteration = native_iteration\n"
            "        return self._required_configuration_iteration\n"
            "\n"
        )
        if "def window_exclusion_configuration_iteration" not in source:
            if source.count(property_anchor) != 1:
                raise ValueError(
                    "Unsupported wc-cuda source layout; no file was changed. "
                    "Missing unique window-exclusion property anchor"
                )
            source = source.replace(
                property_anchor,
                property_source + property_anchor,
                1,
            )
        iteration_anchor = (
            "                    if int(getattr(gpu_frame, 'configuration_iteration', 0)) "
            "< self._required_configuration_iteration:\n"
        )
        iteration_source = (
            "                    native_value = getattr(self._inner, "
            "'required_configuration_iteration', 0)\n"
            "                    native_iteration = int(native_value() if callable(native_value) else native_value) or 0\n"
            "                    if native_iteration > self._required_configuration_iteration:\n"
            "                        self._required_configuration_iteration = native_iteration\n"
        )
        if "native_iteration = int(getattr(self._inner" not in source:
            if source.count(iteration_anchor) != 1:
                raise ValueError(
                    "Unsupported wc-cuda source layout; no file was changed. "
                    "Missing unique configuration-iteration gate"
                )
            source = source.replace(
                iteration_anchor,
                iteration_source + iteration_anchor,
                1,
            )
    if "# D2S: global capture fallback." not in source:
        if not source.endswith(("\n", "\r")):
            source += "\n"
        source += "\n" + _WINDOW_EXCLUSION_HELPERS + "\n"
    compile(source, "wc_cuda/__init__.py", "exec")
    return source


def main() -> int:
    if sys.platform != "win32":
        return 0
    from wc_capture_build import build_native_capture

    runtime = Path(sys.executable).resolve().parent
    bundled_root = Path(__file__).resolve().parents[1]
    if runtime.parent != bundled_root or runtime.name not in {"python3", "python3_cuda"}:
        raise RuntimeError("Run this patch using src/python3 or src/python3_cuda")
    target = runtime / "Lib/site-packages/wc_cuda/__init__.py"
    source = target.read_text(encoding="utf-8")
    patched = patched_source(source)
    native_capture_built = build_native_capture(
        runtime,
        crate_name="wc_cuda",
        package_name="wc_cuda",
        module_name="_wc_cuda",
    )
    if patched != source:
        target.write_text(patched, encoding="utf-8", newline="")
    print(
        "[wc-cuda] GPU capture ready; "
        f"WGC window exclusions {'enabled' if native_capture_built else 'unavailable'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
