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
to a frame. GPU-only sampling is used by the viewer; CPU sampling is opt-in.
Set ``D2S_SYS_LOAD=0`` to disable sampling.

On Windows, built-in PDH GPU counters are filtered to the viewer's adapter
LUID. Dedicated VRAM usage comes from PDH and its capacity from DXGI.
CPU-side keys remain macOS-only.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
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


# Windows PDH GPU counters (registered once, sampled on the background thread).
_PDH_FMT_DOUBLE = 0x00000200
_PDH_MORE_DATA = 0x800007D2 - (1 << 32)
_PDH_GPU_ENGINE_PATH = r"\GPU Engine(*)\Utilization Percentage"
_PDH_VRAM_USED_PATH = r"\GPU Adapter Memory(*)\Dedicated Usage"
_ENGINE_NAME_MAP = {
    "3d": "gpu_3d",
    "copy": "gpu_copy",
    "videodecode": "gpu_decode",
    "videoencode": "gpu_encode",
    "compute": "gpu_compute",
}
_LUID_RE = re.compile(r"luid_0x([0-9a-fA-F]+)_0x([0-9a-fA-F]+)")


class _PdhFmtCounterValue(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_uint32), ("doubleValue", ctypes.c_double)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhFmtCounterValue)]


def _instance_luid(name: str) -> int | None:
    """Parse ``luid_0xHIGH_0xLOW`` from a PDH GPU instance name."""
    match = _LUID_RE.search(name or "")
    if not match:
        return None
    return (int(match.group(1), 16) << 32) | int(match.group(2), 16)


class _DxgiGuid(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _DxgiLuid(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DxgiAdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _DxgiLuid),
        ("Flags", ctypes.c_uint32),
    ]


_IID_IDXGIFACTORY1 = _DxgiGuid(
    0x770AAE78, 0xF26F, 0x4DBA,
    (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)


def _com_vtable(obj):
    return ctypes.cast(
        ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.POINTER(ctypes.c_void_p),
    )


def _query_dxgi_adapter_vram() -> dict[int, int]:
    """Read dedicated VRAM capacity by adapter LUID without a GPU dependency."""
    if sys.platform != "win32":
        return {}
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
        create = dxgi.CreateDXGIFactory1
        create.argtypes = [ctypes.POINTER(_DxgiGuid), ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int32
        factory = ctypes.c_void_p()
        if int(create(ctypes.byref(_IID_IDXGIFACTORY1), ctypes.byref(factory))) != 0:
            return {}
        table = _com_vtable(factory)
        enum_adapters1 = ctypes.WINFUNCTYPE(
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
        )(table[12])
        release = ctypes.WINFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(table[2])
        totals: dict[int, int] = {}
        try:
            index = 0
            while True:
                adapter = ctypes.c_void_p()
                if int(enum_adapters1(factory, index, ctypes.byref(adapter))) != 0 or not adapter:
                    break
                adapter_table = _com_vtable(adapter)
                adapter_release = ctypes.WINFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)(adapter_table[2])
                try:
                    get_desc1 = ctypes.WINFUNCTYPE(
                        ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(_DxgiAdapterDesc1),
                    )(adapter_table[10])
                    desc = _DxgiAdapterDesc1()
                    if int(get_desc1(adapter, ctypes.byref(desc))) == 0 and desc.DedicatedVideoMemory:
                        luid = ((int(desc.AdapterLuid.HighPart) & 0xFFFFFFFF) << 32) | int(desc.AdapterLuid.LowPart)
                        totals[luid] = int(desc.DedicatedVideoMemory)
                finally:
                    adapter_release(adapter)
                index += 1
        finally:
            release(factory)
        return totals
    except (AttributeError, OSError, ValueError):
        return {}


class _WindowsPdhGpuSampler:
    """Sample per-engine GPU utilization and dedicated VRAM via native PDH."""

    def __init__(self) -> None:
        self._pdh = None
        self._query = None
        self._engine_counter = None
        self._vram_used_counter = None
        self._adapter_luid = 0
        self._adapter_vram_bytes: dict[int, int] = {}
        self._last_hr = 0
        self._ok = False
        self._warned = False

    def set_adapter_luid(self, luid: int) -> None:
        self._adapter_luid = int(luid or 0)

    def _warn(self, hr: int) -> None:
        if not self._warned:
            self._warned = True
            print(f"[SysLoad] Windows GPU counters unavailable (hr=0x{int(hr) & 0xFFFFFFFF:08X})", flush=True)

    def _add_counter(self, path: str):
        counter = ctypes.c_void_p()
        hr = int(self._pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(counter)))
        self._last_hr = hr
        return counter if hr == 0 else None

    def start(self) -> bool:
        try:
            pdh = ctypes.WinDLL("pdh.dll")
        except (AttributeError, OSError) as exc:
            self._warn(getattr(exc, "winerror", 0) or 0)
            return False
        pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        pdh.PdhOpenQueryW.restype = ctypes.c_int32
        pdh.PdhAddEnglishCounterW.argtypes = [
            ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
        ]
        pdh.PdhAddEnglishCounterW.restype = ctypes.c_int32
        pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
        pdh.PdhCollectQueryData.restype = ctypes.c_int32
        pdh.PdhGetFormattedCounterArrayW.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(_PdhItem),
        ]
        pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_int32
        pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
        pdh.PdhCloseQuery.restype = ctypes.c_int32
        self._pdh = pdh
        query = ctypes.c_void_p()
        hr = int(pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)))
        if hr != 0:
            self._warn(hr)
            return False
        self._query = query
        self._engine_counter = self._add_counter(_PDH_GPU_ENGINE_PATH)
        if self._engine_counter is None:
            self._warn(self._last_hr)
            self.close()
            return False
        self._vram_used_counter = self._add_counter(_PDH_VRAM_USED_PATH)
        self._adapter_vram_bytes = _query_dxgi_adapter_vram()
        pdh.PdhCollectQueryData(self._query)
        self._ok = True
        return True

    def close(self) -> None:
        if self._pdh is not None and self._query is not None:
            self._pdh.PdhCloseQuery(self._query)
        self._query = None
        self._ok = False

    def _read_counter(self, counter) -> list[tuple[str, float]]:
        if counter is None:
            return []
        size = ctypes.c_uint32(0)
        count = ctypes.c_uint32(0)
        hr = int(self._pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), None,
        ))
        if hr != _PDH_MORE_DATA or not size.value:
            return []
        buffer = ctypes.create_string_buffer(size.value)
        items = ctypes.cast(buffer, ctypes.POINTER(_PdhItem))
        hr = int(self._pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), items,
        ))
        if hr != 0:
            return []
        # Both VALID_DATA (0) and NEW_DATA (1) are successful PDH statuses.
        return [
            (items[i].szName or "", float(items[i].FmtValue.doubleValue))
            for i in range(int(count.value)) if int(items[i].FmtValue.CStatus) in {0, 1}
        ]

    def sample(self) -> dict[str, float | str]:
        if not self._ok or self._query is None:
            return {}
        if int(self._pdh.PdhCollectQueryData(self._query)) != 0:
            return {}
        by_adapter: dict[int, dict[str, float]] = {}
        for name, value in self._read_counter(self._engine_counter):
            luid = _instance_luid(name)
            engine = _ENGINE_NAME_MAP.get(name.rsplit("_engtype_", 1)[-1].strip().lower())
            if luid is None or engine is None:
                continue
            bucket = by_adapter.setdefault(luid, {})
            bucket[engine] = bucket.get(engine, 0.0) + float(value)
        chosen = self._adapter_luid
        if not chosen and by_adapter:
            chosen = max(by_adapter, key=lambda luid: max(by_adapter[luid].values(), default=0.0))
        result: dict[str, float | str] = {}
        engines = by_adapter.get(chosen, {})
        if engines:
            result.update({key: max(0.0, min(100.0, value)) for key, value in engines.items()})
            result["gpu"] = max(result.values(), default=0.0)
            if not self._adapter_luid:
                result["gpu_adapter"] = "auto"
        used_by: dict[int, float] = {}
        for name, value in self._read_counter(self._vram_used_counter):
            luid = _instance_luid(name)
            if luid is not None:
                used_by[luid] = used_by.get(luid, 0.0) + float(value)
        # Never mix another GPU's VRAM into a viewer-bound utilization sample.
        if not chosen and used_by:
            chosen = max(used_by, key=used_by.get)
            result["gpu_adapter"] = "auto"
        if chosen in used_by:
            result["gpu_vram_used_mb"] = used_by[chosen] / (1024.0 * 1024.0)
        total = self._adapter_vram_bytes.get(chosen)
        if total:
            result["gpu_vram_total_mb"] = float(total) / (1024.0 * 1024.0)
        return result


class _HostCpuLoadInfo(ctypes.Structure):
    _fields_ = [("cpu_ticks", ctypes.c_uint32 * _CPU_STATE_MAX)]


class _MachTimebaseInfo(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


class SystemLoadSampler:
    """Sampled machine load, published as a plain dict.

    ``cpu=False`` keeps only the GPU counters: a single IOKit property read on
    macOS or one PDH collection on Windows. CPU sampling is macOS-only and
    opt-in because it walks the process table and shells out to ``ps``.
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
        self._windows_gpu_sampler: _WindowsPdhGpuSampler | None = None
        self._windows_gpu_samples: list[dict[str, float | str]] = []
        self._adapter_luid = 0
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
        if sys.platform == "win32":
            sampler = _WindowsPdhGpuSampler()
            if sampler.start():
                with self._lock:
                    sampler.set_adapter_luid(self._adapter_luid)
                    self._windows_gpu_sampler = sampler
                    self._gpu_ok = True
            return
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
        if sys.platform != "darwin" or not self._gpu_ok:
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

        if sys.platform == "win32":
            samples = self._windows_gpu_samples
            self._windows_gpu_samples = []
            if samples:
                snapshot.update(samples[-1])
                peaks = [float(item["gpu"]) for item in samples if "gpu" in item]
                if peaks:
                    snapshot["gpu_max"] = max(peaks)
        else:
            samples = self._gpu_samples
            self._gpu_samples = []
            device = [s["Device Utilization %"] for s in samples if "Device Utilization %" in s]
            if device:
                snapshot["gpu"] = sum(device) / len(device)
                snapshot["gpu_max"] = float(max(device))

        cpu_enabled = self._cpu_enabled and sys.platform == "darwin"
        system = self._system_cpu() if cpu_enabled else None
        if system is not None:
            snapshot["sys_busy"] = system

        if cpu_enabled:
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

    def set_adapter_luid(self, luid: int) -> None:
        """Restrict Windows GPU counters to the presentation adapter."""
        with self._lock:
            self._adapter_luid = int(luid or 0)
            if self._windows_gpu_sampler is not None:
                self._windows_gpu_sampler.set_adapter_luid(self._adapter_luid)

    def _run(self) -> None:
        self._init_gpu()
        # Prime the CPU baselines so the first published window has deltas.
        try:
            if self._cpu_enabled and sys.platform == "darwin":
                self._system_cpu()
                self._process_cpu()
            self._gpu_stats()
        except Exception:
            pass
        deadline = time.monotonic() + self._interval
        try:
            while not self._stop.wait(_GPU_TICK):
                try:
                    if sys.platform == "win32":
                        sampler = self._windows_gpu_sampler
                        stats = sampler.sample() if sampler is not None else {}
                        if stats:
                            self._windows_gpu_samples.append(stats)
                    else:
                        stats = self._gpu_stats()
                        if stats:
                            self._gpu_samples.append(stats)
                    if time.monotonic() >= deadline:
                        deadline = time.monotonic() + self._interval
                        self._publish()
                except Exception:
                    deadline = time.monotonic() + self._interval
                    continue
        finally:
            if self._windows_gpu_sampler is not None:
                self._windows_gpu_sampler.close()

    # ---------------------------------------------------------------- public
    def start(self) -> bool:
        if sys.platform not in {"darwin", "win32"}:
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
        ("gpu_3d", "gpu_3d", "{:.0f}"),
        ("gpu_copy", "gpu_copy", "{:.0f}"),
        ("gpu_decode", "gpu_decode", "{:.0f}"),
        ("gpu_encode", "gpu_encode", "{:.0f}"),
        ("gpu_compute", "gpu_compute", "{:.0f}"),
        ("gpu_vram_used_mb", "gpu_vram_used_mb", "{:.0f}"),
        ("gpu_vram_total_mb", "gpu_vram_total_mb", "{:.0f}"),
        ("ws_busy", "ws_busy", "{:.1f}"),
        ("app_busy", "app_busy", "{:.1f}"),
        ("browser_busy", "browser_busy", "{:.1f}"),
    ):
        value = snapshot.get(key)
        if isinstance(value, (int, float)):
            fields.append(f"{label}=" + fmt.format(float(value)))
    adapter = snapshot.get("gpu_adapter")
    if adapter:
        fields.append(f"gpu_adapter={adapter}")
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
