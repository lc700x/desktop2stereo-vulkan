"""Honor wc-rocm's capture interval before D3D11-to-HIP transfer.

Apply this patch only to the ROCm Python runtime bundled with Desktop2Stereo.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patched_source(source: str) -> str:
    newline = "\r\n" if "\r\n" in source else "\n"
    changes = (
        (
            f"        self._stream        = None{newline}",
            f"        self._stream        = None{newline}"
            f"        self._minimum_update_interval_s = max(0.0, float(minimum_update_interval or 0) / 1000.0){newline}",
        ),
        (
            f"            while self._running:{newline}"
            f"                res = self._inner.get_frame(self._last_id, timeout=0.1){newline}",
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
            f"                res = self._inner.get_frame(self._last_id, timeout=0.1){newline}",
        ),
    )
    for old, new in changes:
        if new in source:
            continue
        if source.count(old) != 1:
            raise ValueError("Unsupported wc-rocm source layout; no file was changed")
        source = source.replace(old, new, 1)
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
    print("[wc-rocm] bundled GPU capture transfer pacing ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
