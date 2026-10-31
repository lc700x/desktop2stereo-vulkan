"""Background sampler for machine-level load, reported next to frame metrics.

The viewer already reports how long each of its own stages took. On a shared
GPU that is not enough to tell "our work got heavier" from "something else took
the machine", which is the question behind a synthesis time that swings between
15ms and 22ms without any change in content.

The sampler runs on its own thread (never on the presenting thread) and keeps a
small snapshot of:

* ``sys_busy`` / ``app_busy`` / ``browser_busy`` - CPU percentages over the
  reporting window, derived from exact per-process CPU-time deltas
  (``proc_pid_rusage``).
* ``gpu`` / ``gpu_max`` - the GPU's own "Device Utilization %" counters from
  IOKit (mean and peak over the window). This is the shared resource the
  warp pass competes for.
* ``ws_busy`` - ``WindowServer`` CPU. It is root owned, so
  ``proc_pid_rusage`` is not readable; the value comes from ``ps`` and is a
  decaying average over roughly a minute, not an instantaneous window.

The presenting thread only reads a dict, so a slow sample can never add latency
to a frame. Sampling is only started with ``D2S_METRICS_VERBOSE=1``; set
``D2S_SYS_LOAD=0`` to disable it even then.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys
import threading
import time

_ENABLED = os.environ.get("D2S_SYS_LOAD", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

_DEFAULT_INTERVAL = 5.0
_GPU_TICK = 1.0
# How often the pid -> process mapping is rebuilt. Each rebuild costs one
# syscall per running process, so keep it well above the sample interval.
_PID_RESCAN_SECONDS = 60.0

# Process names that usually own the captured content. Substring match so helper
# processes ("Helium Helper (GPU)", "Google Chrome Helper (Renderer)") are
# included in the total.
_BROWSER_NAMES = (
    "Helium",
    "Google Chrome",
    "Chromium",
    "Safari",
    "firefox",
    "Firefox",
    "Microsoft Edge",
    "Arc",
    "哔哩哔哩",
)
# Names that contain a browser tag but are not the browser itself.
_BROWSER_SKIP = ("crashpad", "SafariPlatformSupport", "SafariSafeBrowsing")

_PROC_ALL_PIDS = 1
_RUSAGE_INFO_V2 = 2
_HOST_CPU_LOAD_INFO = 3
_CPU_STATE_MAX = 4
# rusage_info_v2 starts with two u64 of uuid; the CPU times follow.
_OFF_USER_TIME = 16
_OFF_SYSTEM_TIME = 24

_GPU_KEYS = ("Device Utilization %", "Tiler Utilization %", "Renderer Utilization %")


class _HostCpuLoadInfo(ctypes.Structure):
    _fields_ = [("cpu_ticks", ctypes.c_uint32 * _CPU_STATE_MAX)]


class _MachTimebaseInfo(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class SystemLoadSampler:
    """Sampled machine load, published as a plain dict.

    ``cpu=False`` keeps only the GPU counters: they come from a single IOKit
    property read (~50µs, no subprocess, no process enumeration), so the GPU
    line can stay on all the time without touching frame pacing. The CPU parts
    are opt-in because they walk the process table and shell out to ``ps``.
    """

    def __init__(self, interval: float = _DEFAULT_INTERVAL, *, cpu: bool = True) -> None:
        self._interval = max(1.0, float(interval))
        self._cpu_enabled = bool(cpu)
        self._lock = threading.Lock()
        self._snapshot: dict[str, float | str] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._app_pid = os.getpid()
        self._proc_ok = False
        self._gpu_ok = False
        self._ns_per_tick = 1.0
        self._last_cpu: dict[str, float] = {}
        self._last_ticks: tuple[float, ...] | None = None
        self._gpu_samples: list[dict[str, int]] = []
        # pid -> (group, name) for the processes worth sampling. Enumerating
        # every pid costs one syscall per process, and each call competes for
        # the interpreter lock with the render thread, so the mapping is
        # rebuilt only every `_PID_RESCAN_SECONDS`.
        self._interesting: dict[int, tuple[str, str]] = {}
        self._interesting_at = 0.0
        self._init_proc()

    # ------------------------------------------------------------- bindings
    def _init_proc(self) -> None:
        if sys.platform != "darwin":
            return
        try:
            self._libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            libc_path = ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib"
            self._libc = ctypes.CDLL(libc_path, use_errno=True)
        except OSError:
            return
        self._libproc.proc_listpids.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self._libproc.proc_listpids.restype = ctypes.c_int
        self._libproc.proc_name.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self._libproc.proc_name.restype = ctypes.c_int
        self._libproc.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._libproc.proc_pid_rusage.restype = ctypes.c_int
        self._proc_ok = True
        # `ri_user_time`/`ri_system_time` are in mach absolute ticks, not
        # nanoseconds; on Apple silicon one tick is 41.67ns (125/3).
        timebase = _MachTimebaseInfo()
        try:
            if self._libc.mach_timebase_info(ctypes.byref(timebase)) == 0 and timebase.denom:
                self._ns_per_tick = timebase.numer / timebase.denom
        except AttributeError:
            pass

    # ------------------------------------------------------------ processes
    def _pids(self) -> list[int]:
        needed = self._libproc.proc_listpids(_PROC_ALL_PIDS, 0, None, 0)
        if needed <= 0:
            return []
        buf = (ctypes.c_int * (needed // 4 + 16))()
        written = self._libproc.proc_listpids(
            _PROC_ALL_PIDS, 0, ctypes.byref(buf), ctypes.sizeof(buf)
        )
        if written <= 0:
            return []
        count = min(len(buf), written // 4)
        return [int(buf[i]) for i in range(count) if int(buf[i]) > 0]

    def _name(self, pid: int) -> str:
        buf = ctypes.create_string_buffer(64)
        if self._libproc.proc_name(pid, buf, ctypes.sizeof(buf)) <= 0:
            return ""
        return buf.value.decode("utf-8", "replace")

    def _cpu_ticks_of(self, pid: int) -> float | None:
        buf = ctypes.create_string_buffer(512)
        if self._libproc.proc_pid_rusage(pid, _RUSAGE_INFO_V2, buf) != 0:
            return None
        raw = buf.raw
        user = int.from_bytes(raw[_OFF_USER_TIME : _OFF_USER_TIME + 8], "little")
        system = int.from_bytes(raw[_OFF_SYSTEM_TIME : _OFF_SYSTEM_TIME + 8], "little")
        return float(user + system)

    def _refresh_interesting(self) -> None:
        """Rebuild the pid -> process mapping (expensive, so kept rare)."""
        interesting: dict[int, tuple[str, str]] = {}
        for index, pid in enumerate(self._pids()):
            if index and index % 64 == 0:
                # Do not monopolise the interpreter lock while enumerating.
                time.sleep(0)
            if pid == self._app_pid:
                interesting[pid] = ("app", "app")
                continue
            name = self._name(pid)
            if not name:
                continue
            tag = _browser_tag(name)
            if tag is None:
                continue
            interesting[pid] = ("browser", tag)
        self._interesting = interesting
        self._interesting_at = time.monotonic()

    def _process_cpu(self) -> tuple[dict[str, float], str]:
        """CPU percentages per interesting process group over the window."""
        now = time.monotonic()
        if not self._interesting or now - self._interesting_at > _PID_RESCAN_SECONDS:
            self._refresh_interesting()
        totals: dict[str, float] = {}
        pid_cpu: dict[str, float] = {}
        browser_name = ""
        browser_rate = -1.0
        dead: list[int] = []
        elapsed = now - self._last_cpu.get("_at", 0.0) if self._last_cpu else 0.0
        for pid, (group, name) in self._interesting.items():
            cpu = self._cpu_ticks_of(pid)
            if cpu is None:
                # Process exited (or is not ours to read); drop it and let the
                # next rescan pick up replacements.
                dead.append(pid)
                continue
            totals[group] = totals.get(group, 0.0) + cpu
            if group != "browser":
                continue
            pid_cpu[f"pid:{pid}"] = cpu
            rate = cpu - self._last_cpu.get(f"pid:{pid}", 0.0)
            if rate > browser_rate:
                browser_rate = rate
                browser_name = name
        for pid in dead:
            self._interesting.pop(pid, None)

        percents: dict[str, float] = {}
        if self._last_cpu and elapsed > 0.5:
            for group, cpu in totals.items():
                previous = self._last_cpu.get(group)
                if previous is None:
                    continue
                used_seconds = (cpu - previous) * self._ns_per_tick / 1e9
                percents[f"{group}_cpu"] = used_seconds / elapsed * 100.0
        totals["_at"] = now
        totals.update(pid_cpu)
        self._last_cpu = totals
        return percents, browser_name

    # -------------------------------------------------------------- system
    def _cpu_ticks(self) -> tuple[float, ...] | None:
        info = _HostCpuLoadInfo()
        count = ctypes.c_uint32(_CPU_STATE_MAX)
        host = self._libc.mach_host_self()
        if (
            self._libc.host_statistics(
                ctypes.c_uint32(host),
                ctypes.c_int(_HOST_CPU_LOAD_INFO),
                ctypes.byref(info),
                ctypes.byref(count),
            )
            != 0
        ):
            return None
        return tuple(float(v) for v in info.cpu_ticks)

    def _system_cpu(self) -> float | None:
        ticks = self._cpu_ticks()
        if ticks is None:
            return None
        previous = self._last_ticks
        self._last_ticks = ticks
        if previous is None:
            return None
        delta = [a - b for a, b in zip(ticks, previous)]
        total = sum(delta)
        if total <= 0:
            return None
        busy = delta[0] + delta[1] + delta[3]
        return busy / total * 100.0

    def _windowserver_cpu(self) -> float | None:
        """WindowServer CPU from ``ps`` (decaying average over ~a minute)."""
        try:
            out = subprocess.run(
                ["ps", "-A", "-o", "pid=,%cpu=,comm="],
                capture_output=True,
                text=True,
                timeout=5.0,
                # posix_spawn instead of fork: the app is multi-threaded and
                # holds GPU driver locks, so forking from this thread is best
                # avoided.
                close_fds=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        for line in out.splitlines():
            line = line.rstrip()
            if not line.endswith("/WindowServer"):
                continue
            fields = line.split(None, 2)
            if len(fields) < 2:
                continue
            try:
                return float(fields[1])
            except ValueError:
                continue
        return None

    # ----------------------------------------------------------------- GPU
    def _init_gpu(self) -> None:
        if sys.platform != "darwin":
            return
        try:
            iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
            cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        except OSError:
            return
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cf.CFDictionaryGetValue.restype = ctypes.c_void_p
        cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        cf.CFNumberGetValue.restype = ctypes.c_int
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceGetMatchingServices.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
        iokit.IOIteratorNext.argtypes = [ctypes.c_uint32]
        iokit.IOIteratorNext.restype = ctypes.c_uint32
        iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]
        iokit.IORegistryEntryCreateCFProperty.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
        self._iokit = iokit
        self._cf = cf
        self._gpu_ok = True

    def _cfstring(self, text: str) -> ctypes.c_void_p:
        # 0x08000100 == kCFStringEncodingUTF8
        return self._cf.CFStringCreateWithCString(None, text.encode("utf-8"), 0x08000100)

    def _gpu_stats(self) -> dict[str, int]:
        if not self._gpu_ok:
            return {}
        cf, iokit = self._cf, self._iokit
        matching = iokit.IOServiceMatching(b"AGXAccelerator")
        if not matching:
            return {}
        iterator = ctypes.c_uint32(0)
        if iokit.IOServiceGetMatchingServices(0, matching, ctypes.byref(iterator)) != 0:
            return {}
        stats: dict[str, int] = {}
        key = self._cfstring("PerformanceStatistics")
        service = iokit.IOIteratorNext(iterator)
        try:
            while service:
                dictionary = iokit.IORegistryEntryCreateCFProperty(service, key, None, 0)
                if dictionary:
                    for name in _GPU_KEYS:
                        sub_key = self._cfstring(name)
                        value = cf.CFDictionaryGetValue(dictionary, sub_key)
                        if value:
                            number = ctypes.c_longlong(0)
                            if cf.CFNumberGetValue(value, 4, ctypes.byref(number)):
                                stats[name] = int(number.value)
                        if sub_key:
                            cf.CFRelease(sub_key)
                    cf.CFRelease(dictionary)
                iokit.IOObjectRelease(service)
                service = iokit.IOIteratorNext(iterator)
        finally:
            iokit.IOObjectRelease(iterator.value)
            if key:
                cf.CFRelease(key)
        return stats

    # ------------------------------------------------------------- sampling
    def _publish(self) -> None:
        snapshot: dict[str, float | str] = {}

        samples = self._gpu_samples
        self._gpu_samples = []
        device = [s["Device Utilization %"] for s in samples if "Device Utilization %" in s]
        if device:
            snapshot["gpu"] = sum(device) / len(device)
            snapshot["gpu_max"] = float(max(device))

        system = self._system_cpu() if self._cpu_enabled else None
        if system is not None:
            snapshot["sys_busy"] = system

        if self._cpu_enabled:
            percents, browser_name = self._process_cpu()
            for group in ("app", "browser"):
                value = percents.get(f"{group}_cpu")
                if value is not None:
                    snapshot[f"{group}_busy"] = max(0.0, value)
            if browser_name and "browser_busy" in snapshot:
                snapshot["browser_name"] = browser_name

            windowserver = self._windowserver_cpu()
            if windowserver is not None:
                snapshot["ws_busy"] = windowserver

        with self._lock:
            self._snapshot = snapshot

    def _run(self) -> None:
        self._init_gpu()
        # Prime the CPU baselines so the first published window has deltas.
        try:
            if self._cpu_enabled:
                self._system_cpu()
                self._process_cpu()
            self._gpu_stats()
        except Exception:
            pass
        deadline = time.monotonic() + self._interval
        while not self._stop.wait(_GPU_TICK):
            try:
                stats = self._gpu_stats()
                if stats:
                    self._gpu_samples.append(stats)
                if time.monotonic() >= deadline:
                    deadline = time.monotonic() + self._interval
                    self._publish()
            except Exception:
                deadline = time.monotonic() + self._interval
                continue

    # ---------------------------------------------------------------- public
    def start(self) -> bool:
        if sys.platform != "darwin":
            return False
        with self._lock:
            if self._thread is not None:
                return True
            thread = threading.Thread(target=self._run, name="d2s-sys-load", daemon=True)
            self._thread = thread
        thread.start()
        return True

    def snapshot(self) -> dict[str, float | str]:
        with self._lock:
            return dict(self._snapshot)


def _browser_tag(name: str) -> str | None:
    """Return the browser a process belongs to, or ``None``.

    Matching is anchored on the executable basename so unrelated processes that
    merely contain a browser name are ignored (e.g. ``ArcZipFinderSync`` must
    not be counted as the Arc browser). Helper processes are included.
    """
    base = name.split("/")[-1].strip()
    if not base or base.startswith("com."):
        return None
    if any(skip in base for skip in _BROWSER_SKIP):
        return None
    for tag in _BROWSER_NAMES:
        if base == tag or base.startswith(f"{tag} Helper") or base.startswith(f"{tag} ("):
            return tag.replace(" ", "_")
    return None


def format_load_fields(snapshot: dict | None) -> list[str]:
    """Render a snapshot as ``key=value`` fields for the metrics line."""
    if not snapshot:
        return []
    fields: list[str] = []
    for key, label, fmt in (
        # "busy" instead of "cpu": the GUI child-log handler promotes any line
        # containing "cpu" to CRITICAL, which would mislabel a metrics line.
        ("sys_busy", "sys_busy", "{:.1f}"),
        ("gpu", "sys_gpu", "{:.0f}"),
        ("gpu_max", "sys_gpu_max", "{:.0f}"),
        ("ws_busy", "ws_busy", "{:.1f}"),
        ("app_busy", "app_busy", "{:.1f}"),
        ("browser_busy", "browser_busy", "{:.1f}"),
    ):
        value = snapshot.get(key)
        if isinstance(value, (int, float)):
            fields.append(f"{label}=" + fmt.format(float(value)))
    name = snapshot.get("browser_name")
    if name and "browser_busy" in snapshot:
        fields.append(f"browser={name}")
    return fields


_sampler: SystemLoadSampler | None = None
_sampler_lock = threading.Lock()


def install_system_load_sampler(
    interval: float = _DEFAULT_INTERVAL, *, cpu: bool = True
) -> SystemLoadSampler | None:
    """Start the machine-load sampler once per process (idempotent).

    ``cpu=False`` samples only the GPU counters (default for the slim metrics
    line); ``cpu=True`` adds the whole-machine, per-process and WindowServer
    numbers used by ``D2S_METRICS_VERBOSE=1``.
    """
    global _sampler
    if not _ENABLED:
        return None
    with _sampler_lock:
        if _sampler is None:
            sampler = SystemLoadSampler(interval, cpu=cpu)
            if not sampler.start():
                return None
            _sampler = sampler
        elif cpu:
            # Upgrade a running GPU-only sampler instead of starting a second
            # thread; the loop reads the flag on every publish.
            _sampler._cpu_enabled = True
        return _sampler
