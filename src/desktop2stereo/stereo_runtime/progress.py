from __future__ import annotations

import json
import operator
import sys
import threading
import time
from contextlib import contextmanager


def progress_write(message: str, *, leading_newline: bool = False) -> None:
    text = str(message)
    if leading_newline:
        text = "\n" + text
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def status_write(message: str) -> None:
    progress_write("[D2S_STATUS] " + str(message))


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{int(value)} B"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def emit_progress_event(desc: str, completed: int, total: int | None, *, started_at: float, unit: str = "bytes") -> None:
    elapsed = max(0.001, time.perf_counter() - started_at)
    percent = (float(completed) / float(total) * 100.0) if total else None
    speed = float(completed) / elapsed
    remaining = ((float(total) - float(completed)) / speed) if total and speed > 0 and completed < total else 0.0
    if unit == "steps":
        downloaded = f"{int(completed)} steps"
        size = f"{int(total)} steps" if total else "unknown"
        speed_text = f"{speed:.1f} steps/s"
    else:
        unit = "bytes"
        downloaded = _format_bytes(completed)
        size = _format_bytes(total)
        speed_text = f"{_format_bytes(speed)}/s"
    payload = {
        "desc": str(desc or "Progress"),
        "completed": int(completed),
        "total": int(total) if total else None,
        "percent": round(min(100.0, max(0.0, percent)), 1) if percent is not None else None,
        "elapsed": _format_duration(elapsed),
        "downloaded": downloaded,
        "size": size,
        "speed": speed_text,
        "eta": _format_duration(remaining) if total else "unknown",
        "unit": unit,
    }
    progress_write("[D2S_PROGRESS] " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


class _NullProgress:
    def __init__(self, desc: str = "", total: int | None = None) -> None:
        self.desc = desc
        self.n = 0
        self.total = total
        self._started_at = time.perf_counter()
        self._last_emit_at = 0.0

    def update(self, amount=1):
        self.n += amount

    def set_description(self, desc):
        self.desc = str(desc or "")

    def set_postfix_str(self, _value, refresh=True):
        return None

    def refresh(self):
        return None

    def close(self):
        return None

    def _emit(self, *, force: bool = False, interval_s: float = 0.25) -> None:
        if not self.total:
            return
        now = time.perf_counter()
        complete = self.n >= self.total
        if not force and not complete and now - self._last_emit_at < interval_s:
            return
        self._last_emit_at = now
        emit_progress_event(self.desc, min(self.n, self.total), self.total, started_at=self._started_at)


class _StageLogProgress(_NullProgress):
    def __init__(self, desc: str, total: int) -> None:
        super().__init__(desc)
        self.total = total
        self._last_postfix = None

    def update(self, amount=1):
        self.n += amount

    def set_postfix_str(self, value, refresh=True):
        postfix = str(value or "").strip()
        if postfix and postfix != self._last_postfix:
            print(f"[Main] {self.desc}: {postfix}", flush=True)
            self._last_postfix = postfix


class DownloadProgress(_NullProgress):
    _lock = threading.RLock()

    @classmethod
    def get_lock(cls):
        return cls._lock

    @classmethod
    def set_lock(cls, lock):
        cls._lock = lock

    def __init__(self, *args, **kwargs):
        iterable = args[0] if args and not isinstance(args[0], (int, float)) else None
        total = kwargs.get("total")
        if total is None:
            total = operator.length_hint(iterable, 0) if iterable is not None else (args[0] if args else None)
        super().__init__(str(kwargs.get("desc") or "download"), total=total)
        self.iterable = iterable
        requested_unit = str(kwargs.get("unit") or "").strip().lower()
        if requested_unit in {"step", "steps", "it", "items"}:
            self._progress_unit = "steps"
        elif requested_unit in {"byte", "bytes", "b"}:
            self._progress_unit = "bytes"
        else:
            # tqdm.contrib.concurrent.thread_map constructs the progress
            # object with total+smoothing but without passing the iterable.
            self._progress_unit = "steps" if iterable is not None or "smoothing" in kwargs else "bytes"
        self.n = int(kwargs.get("initial") or 0)
        self._mininterval = float(kwargs.get("mininterval") or 0.25)
        self.leave = bool(kwargs.get("leave", True))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @property
    def fp(self):
        return sys.stdout

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def update(self, amount=1):
        self.n += amount
        self._emit(interval_s=self._mininterval)
        return None

    def _emit(self, *, force: bool = False, interval_s: float = 0.25) -> None:
        if not self.total:
            return
        now = time.perf_counter()
        complete = self.n >= self.total
        if not force and not complete and now - self._last_emit_at < interval_s:
            return
        self._last_emit_at = now
        emit_progress_event(
            self.desc,
            min(self.n, self.total),
            self.total,
            started_at=self._started_at,
            unit=self._progress_unit,
        )

    def set_description(self, desc, refresh=True):
        self.desc = str(desc or "")
        return None

    def set_postfix_str(self, value, refresh=True):
        return _StageLogProgress.set_postfix_str(self, value, refresh=refresh)

    def refresh(self):
        self._emit(force=True)
        return None

    def close(self):
        self._emit(force=True)
        return None


@contextmanager
def activity_progress(desc: str, *, interval_s: float = 0.2):
    """Report activity and elapsed time for operations without percent callbacks."""
    print(f"[Main] {desc}...", flush=True)
    stop_event = threading.Event()
    progress = _NullProgress(desc)

    def _emit(state: str = "running") -> None:
        payload = {
            "desc": progress.desc,
            "percent": 100.0 if state == "complete" else None,
            "elapsed": _format_duration(time.perf_counter() - progress._started_at),
            "unit": "activity",
            "state": state,
        }
        progress_write("[D2S_PROGRESS] " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    _emit()

    def _pulse() -> None:
        while not stop_event.wait(interval_s):
            progress.update(1)
            _emit()

    thread = threading.Thread(target=_pulse, name=f"Progress:{desc}", daemon=True)
    thread.start()
    complete = False
    try:
        yield progress
        complete = True
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        _emit("complete" if complete else "failed")


@contextmanager
def stage_progress(desc: str, total: int):
    yield _StageLogProgress(desc, total)

@contextmanager
def file_size_progress(desc: str, path, *, total_bytes: int, interval_s: float = 0.2):
    """Track a file's byte size as an approximate progress bar."""
    from pathlib import Path

    target = Path(path)
    total = max(1, int(total_bytes or 1))
    stop_event = threading.Event()
    progress = _NullProgress(desc, total)
    last_size = 0

    def _poll() -> None:
        nonlocal last_size
        while not stop_event.wait(interval_s):
            try:
                size = target.stat().st_size if target.exists() else 0
            except OSError:
                size = last_size
            size = min(size, total)
            progress.n = size
            progress._emit()
            last_size = size

    thread = threading.Thread(target=_poll, name=f"Progress:{desc}", daemon=True)
    thread.start()
    try:
        yield progress
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
        try:
            size = target.stat().st_size if target.exists() else last_size
        except OSError:
            size = last_size
        size = min(size, total)
        progress.n = size
        progress._emit(force=True)


def write_bytes_with_progress(path, data, desc: str, *, chunk_size: int = 8 * 1024 * 1024):
    """Write bytes with a real byte-count progress bar."""
    from pathlib import Path

    target = Path(path)
    try:
        blob = memoryview(data)
    except TypeError:
        blob = memoryview(bytes(data))
    total = len(blob)
    progress = DownloadProgress(total=total, desc=desc)
    with target.open("wb") as file:
        for offset in range(0, total, chunk_size):
            chunk = blob[offset:offset + chunk_size]
            file.write(chunk)
            progress.update(len(chunk))
    progress.close()
