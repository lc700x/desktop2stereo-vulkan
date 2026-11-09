from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

import utils.sys_load as sys_load


def test_pdh_luid_parser_preserves_both_halves() -> None:
    assert sys_load._instance_luid("pid_42_luid_0xFEDCBA98_0x76543210_eng_0_engtype_3D") == 0xFEDCBA9876543210
    assert sys_load._instance_luid("unknown") is None


def test_pdh_array_accepts_valid_and_new_data_and_skips_errors() -> None:
    rows = (sys_load._PdhItem * 3)()
    for index, status in enumerate((0, 1, 0xC0000BC6)):
        rows[index].szName = f"engine{index}"
        rows[index].FmtValue.CStatus = status
        rows[index].FmtValue.doubleValue = 10.0 + index

    def read_array(counter, flags, size, count, destination):
        ctypes.cast(size, ctypes.POINTER(ctypes.c_uint32))[0] = ctypes.sizeof(rows)
        ctypes.cast(count, ctypes.POINTER(ctypes.c_uint32))[0] = len(rows)
        if destination is None:
            return sys_load._PDH_MORE_DATA
        ctypes.memmove(destination, rows, ctypes.sizeof(rows))
        return 0

    sampler = sys_load._WindowsPdhGpuSampler()
    sampler._pdh = SimpleNamespace(PdhGetFormattedCounterArrayW=read_array)
    assert sampler._read_counter(object()) == [("engine0", 10.0), ("engine1", 11.0)]
    assert sampler._read_counter(None) == []


@pytest.mark.parametrize("selected", [2, 3, 0])
def test_windows_gpu_and_vram_use_the_same_adapter(monkeypatch, selected) -> None:
    sampler = sys_load._WindowsPdhGpuSampler()
    sampler._ok = True
    sampler._query = object()
    sampler._engine_counter = object()
    sampler._vram_used_counter = object()
    sampler._adapter_vram_bytes = {1: 16384 * 1024**2, 2: 8192 * 1024**2}
    sampler.set_adapter_luid(selected)
    sampler._pdh = SimpleNamespace(PdhCollectQueryData=lambda query: 0)
    counters = {
        sampler._engine_counter: [
            ("pid_1_luid_0x0_0x1_eng_0_engtype_3D", 99.0),
            ("pid_2_luid_0x0_0x2_eng_0_engtype_3D", 30.0),
            ("pid_3_luid_0x0_0x2_eng_0_engtype_3D", 15.0),
            ("pid_2_luid_0x0_0x2_eng_1_engtype_VideoEncode", 8.0),
        ],
        sampler._vram_used_counter: [
            ("luid_0x0_0x1_phys_0", 4096 * 1024**2),
            ("luid_0x0_0x2_phys_0", 1024 * 1024**2),
        ],
    }
    monkeypatch.setattr(sampler, "_read_counter", lambda counter: counters[counter])
    result = sampler.sample()
    if selected == 2:
        assert result == {
            "gpu_3d": 45.0, "gpu_encode": 8.0, "gpu": 45.0,
            "gpu_vram_used_mb": 1024.0, "gpu_vram_total_mb": 8192.0,
        }
    elif selected == 3:
        assert result == {}  # A missing selected GPU must not silently use another GPU.
    else:
        assert result["gpu"] == 99.0
        assert result["gpu_vram_used_mb"] == 4096.0
        assert result["gpu_vram_total_mb"] == 16384.0
        assert result["gpu_adapter"] == "auto"


def test_windows_sampler_publishes_gpu_without_macos_cpu_calls(monkeypatch) -> None:
    monkeypatch.setattr(sys_load.sys, "platform", "win32")
    sampler = sys_load.SystemLoadSampler(cpu=True)
    sampler._windows_gpu_samples = [{"gpu": 70.0}, {"gpu": 42.0, "gpu_encode": 8.0}]
    monkeypatch.setattr(sampler, "_system_cpu", lambda: pytest.fail("macOS CPU query on Windows"))
    sampler._publish()
    assert sampler.snapshot() == {"gpu": 42.0, "gpu_encode": 8.0, "gpu_max": 70.0}
    assert sampler._windows_gpu_samples == []


def test_macos_gpu_publication_keeps_mean_and_peak(monkeypatch) -> None:
    monkeypatch.setattr(sys_load.sys, "platform", "darwin")
    sampler = sys_load.SystemLoadSampler(cpu=False)
    sampler._gpu_samples = [{"Device Utilization %": 20}, {"Device Utilization %": 40}]
    sampler._publish()
    assert sampler.snapshot() == {"gpu": 30.0, "gpu_max": 40.0}


def test_windows_gpu_counter_unavailable_is_nonfatal(monkeypatch, capsys) -> None:
    def unavailable(name):
        raise OSError("PDH unavailable")

    monkeypatch.setattr(ctypes, "WinDLL", unavailable, raising=False)
    sampler = sys_load._WindowsPdhGpuSampler()
    assert sampler.start() is False
    assert sampler.sample() == {}
    assert "Windows GPU counters unavailable" in capsys.readouterr().out


def test_gpu_adapter_binding_survives_background_initialization(monkeypatch) -> None:
    monkeypatch.setattr(sys_load.sys, "platform", "win32")
    sampler = sys_load.SystemLoadSampler(cpu=False)
    sampler.set_adapter_luid(0x1234)
    native = sys_load._WindowsPdhGpuSampler()
    monkeypatch.setattr(native, "start", lambda: True)
    monkeypatch.setattr(sys_load, "_WindowsPdhGpuSampler", lambda: native)
    sampler._init_gpu()
    assert native._adapter_luid == 0x1234
    sampler.set_adapter_luid(0x5678)
    assert native._adapter_luid == 0x5678
