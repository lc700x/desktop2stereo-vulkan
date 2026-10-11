from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

import stereo_runtime.triton_runtime as triton_runtime


def test_cached_triton_probe_skips_windows_compiler_discovery(monkeypatch) -> None:
    cached = triton_runtime.TritonRuntimeInfo(
        available=False,
        vendor="unknown",
        backend="unavailable",
        reason="cached",
        device="cpu",
        triton_version="test",
        torch_version="test",
        is_windows=True,
    )
    monkeypatch.setitem(triton_runtime._CACHE, "cpu", cached)
    monkeypatch.setattr(
        triton_runtime,
        "_ensure_windows_cc",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected compiler lookup")),
    )

    assert triton_runtime.probe_triton_runtime(torch.device("cpu")) is cached


def test_windows_compiler_discovery_is_cached_and_can_be_forced(monkeypatch) -> None:
    calls = []
    fake_os = SimpleNamespace(name="nt", environ={"CC": ""}, path=os.path)
    monkeypatch.setattr(triton_runtime, "os", fake_os)
    monkeypatch.setattr(triton_runtime, "_WINDOWS_CC_CHECKED", False)
    monkeypatch.setitem(
        sys.modules,
        "triton.windows_utils",
        SimpleNamespace(find_msvc=lambda **_kwargs: calls.append("probe") or ("msvc", None, None)),
    )

    triton_runtime._ensure_windows_cc()
    triton_runtime._ensure_windows_cc()
    assert calls == ["probe"]

    triton_runtime._ensure_windows_cc(force=True)
    assert calls == ["probe", "probe"]

    triton_runtime.clear_triton_probe_cache()
    triton_runtime._ensure_windows_cc()
    assert calls == ["probe", "probe", "probe"]
