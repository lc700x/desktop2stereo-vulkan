"""Depth model loading must work offline from the local HuggingFace cache.

Regression: the endpoint probe (_reachable_hf_endpoints) hard-fails when
huggingface.co / hf-mirror.com are unreachable (DNS down, no VPN), which
killed startup and RTMP calibration even though the weights were already
fully cached. Loading must prefer the local cache with zero network I/O.
"""
from __future__ import annotations

import pytest

from stereo_runtime import depth_provider


def test_load_hf_prefers_local_cache_and_skips_network_probe(monkeypatch) -> None:
    def fail_if_probed(*_args, **_kwargs):
        raise AssertionError("network endpoint probe must not run for a cached model")

    monkeypatch.setattr(depth_provider, "_reachable_hf_endpoints", fail_if_probed)
    monkeypatch.setattr(
        depth_provider, "_hf_endpoint_candidates", fail_if_probed
    )

    result = depth_provider._load_hf_with_endpoint_fallback(
        lambda model_id: f"online:{model_id}",
        "org/model",
        local_first=lambda: "cached-model",
    )

    assert result == "cached-model"


def test_load_hf_falls_through_to_online_when_cache_incomplete(monkeypatch) -> None:
    calls = []

    def fake_endpoints(model_id):
        calls.append(model_id)
        return ("https://hf-mirror.com",)

    monkeypatch.setattr(depth_provider, "_reachable_hf_endpoints", fake_endpoints)
    monkeypatch.setattr(depth_provider, "_probe_download_url", lambda *a, **k: True)
    monkeypatch.setattr(
        depth_provider,
        "_hf_endpoint",
        lambda endpoint: pytest.importorskip("contextlib").nullcontext(),
    )
    monkeypatch.setattr(
        depth_provider,
        "_hf_download_progress_patch",
        lambda: pytest.importorskip("contextlib").nullcontext(),
    )

    def local_first():
        raise RuntimeError("cache incomplete")

    result = depth_provider._load_hf_with_endpoint_fallback(
        lambda model_id: f"online:{model_id}",
        "org/model",
        local_first=local_first,
    )

    assert result == "online:org/model"
    assert calls == ["org/model"]


def test_load_hf_without_local_first_keeps_online_path(monkeypatch) -> None:
    calls = []

    def fake_endpoints(model_id):
        calls.append(model_id)
        return ("https://huggingface.co",)

    monkeypatch.setattr(depth_provider, "_reachable_hf_endpoints", fake_endpoints)
    monkeypatch.setattr(depth_provider, "_probe_download_url", lambda *a, **k: True)
    monkeypatch.setattr(
        depth_provider,
        "_hf_endpoint",
        lambda endpoint: pytest.importorskip("contextlib").nullcontext(),
    )
    monkeypatch.setattr(
        depth_provider,
        "_hf_download_progress_patch",
        lambda: pytest.importorskip("contextlib").nullcontext(),
    )

    result = depth_provider._load_hf_with_endpoint_fallback(
        lambda model_id: f"online:{model_id}",
        "org/model",
    )

    assert result == "online:org/model"
    assert calls == ["org/model"]


def test_torch_provider_local_files_only_never_probes_network(monkeypatch) -> None:
    import transformers

    monkeypatch.setattr(
        depth_provider, "_load_hf_with_endpoint_fallback",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("local_files_only must bypass the endpoint fallback")
        ),
    )
    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    fake_model = FakeModel()
    monkeypatch.setattr(
        transformers.AutoModelForDepthEstimation,
        "from_pretrained",
        lambda model_id, **kwargs: fake_model,
    )

    provider = depth_provider.DistillAnyDepthBase518(
        device="cpu",
        cache_dir="/tmp/d2s-test-cache",
        local_files_only=True,
    )
    loaded = provider.load()

    assert loaded is fake_model
