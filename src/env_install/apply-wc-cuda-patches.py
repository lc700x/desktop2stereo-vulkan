"""Honor wc-cuda's existing interval before D3D11/CUDA transfer.

The 0.1.3 Python wrapper accepts minimum_update_interval but ignores it.
Apply only to the Python runtime bundled beside this installer.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patched_source(source: str) -> str:
    changes = (
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
        if new in source:
            continue
        if source.count(old) != 1:
            raise ValueError("Unsupported wc-cuda source layout; no file was changed")
        source = source.replace(old, new, 1)
    compile(source, "wc_cuda/__init__.py", "exec")
    return source


def main() -> int:
    if sys.platform != "win32":
        return 0
    runtime = Path(sys.executable).resolve().parent
    bundled_root = Path(__file__).resolve().parents[1]
    if runtime.parent != bundled_root or runtime.name not in {"python3", "python3_cuda"}:
        raise RuntimeError("Run this patch using src/python3 or src/python3_cuda")
    target = runtime / "Lib/site-packages/wc_cuda/__init__.py"
    source = target.read_text(encoding="utf-8")
    patched = patched_source(source)
    if patched != source:
        target.write_text(patched, encoding="utf-8", newline="")
    print("[wc-cuda] GPU capture transfer pacing ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
