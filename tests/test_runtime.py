import os
import sys
from pathlib import Path

from path_config import APP_ROOT

import logging
import pytest
import torch
import stereo_runtime.runtime as runtime_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from stereo_runtime import StereoRuntime, StereoRuntimeConfig
from stereo_runtime.depth_provider import (
    DepthProfileResult,
    DepthProviderInfo,
    DownloadProgress,
    _RESOLVED_HF_MODEL_FILES,
    _hf_endpoint_candidates,
    _hf_download_progress_patch,
    _load_hf_with_endpoint_fallback,
    _probe_download_url,
    _reachable_hf_endpoints,
    _hf_resolve_url,
    _print_download_preparing_progress,
    _progress_print,
    _raise_model_resolution_error,
    _resolve_hf_model_file,
)
from stereo_runtime.runtime import RollingRuntimeStats
from stereo_runtime.synthesis import StereoConfig


class FakeDepthProvider:
    def __init__(self) -> None:
        self.load_count = 0
        self.predict_count = 0
        self.close_count = 0
        self.info = DepthProviderInfo(
            provider="fake",
            model_name="fake-depth",
            model_id="fake/model",
            depth_resolution=2,
            cache_dir=".",
            depth_backend="fake",
            runtime="test",
        )

    def load(self) -> None:
        self.load_count += 1

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        self.predict_count += 1
        b, _, h, w = rgb.shape
        depth = torch.linspace(0, 1, w, dtype=rgb.dtype, device=rgb.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return DepthProfileResult(depth=depth, preprocess_ms=1.0, model_ms=2.0, postprocess_ms=3.0)

    def close(self) -> None:
        self.close_count += 1


def test_runtime_process_rgb_frame_uses_persistent_provider_and_returns_report():
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        output_format="half_sbs",
        temporal=False,
        max_disparity_px=18.0,
        parallax_preset="comfort",
    )
    runtime = StereoRuntime(config, depth_provider=provider, stats_window=4, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 8, 12)

    first = runtime.process_rgb_frame(rgb)
    second = runtime.process_rgb_frame(rgb)

    assert provider.load_count == 1
    assert provider.predict_count == 2
    assert first.depth.shape == (1, 1, 8, 12)
    assert first.left_eye.shape == rgb.shape
    assert first.right_eye.shape == rgb.shape
    assert first.sbs.shape == rgb.shape
    assert first.timing["depth_preprocess_ms"] == 1.0
    assert first.timing["depth_model_ms"] == 2.0
    assert first.timing["depth_postprocess_ms"] == 3.0
    assert first.timing["synthesis_ms"] >= 0.0
    assert first.debug_info["runtime_depth_backend"] == "pytorch_cuda"
    assert first.debug_info["runtime_output_format"] == "half_sbs"
    assert first.debug_info["runtime_quality_mode"] == "fast"
    assert first.debug_info["stereo_synthesis_mode"] == "packed_synthesis"
    assert first.debug_info["motion_depth_reason"] == "disabled"
    assert first.debug_info["output_format"] == "half_sbs"
    assert first.debug_info["max_disparity_px"] == 18.0
    assert first.debug_info["parallax_preset"] == "comfort"
    assert first.debug_info["depth_provider_size"] == "2x2"
    assert first.debug_info["depth_render_size"] == "12x8"
    assert first.provider_info["provider"] == "fake"
    assert second.timing["total_ms"] >= 0.0

    report = runtime.to_report()
    assert report["depth_backend_resolved"] == "fake"
    assert report["stereo_backend"] == "fast"
    assert report["depth_provider"]["provider"] == "fake"
    assert "last_timing" in report
    assert report["rolling_stats"]["count"] == 2
    assert report["rolling_stats"]["stages"]["depth_model_ms"]["mean"] == 2.0
    assert report["rolling_stats"]["fps"]["latest"] > 0.0

    runtime.close()
    assert provider.close_count == 1


def test_runtime_process_rgb_frame_accepts_precomputed_depth_profile():
    provider = FakeDepthProvider()
    runtime = StereoRuntime(
        StereoRuntimeConfig(
            model_id="lc700x/Distill-Any-Depth-Base-hf",
            depth_backend="pytorch_cuda",
            stereo_quality="fast",
            output_format="half_sbs",
            temporal=False,
        ),
        depth_provider=provider,
        collect_memory_stats=False,
    )
    rgb = torch.rand(1, 3, 8, 12)
    profile = provider.predict_profile(rgb)

    result = runtime.process_rgb_frame(rgb, depth_profile=profile)

    assert provider.predict_count == 1
    assert result.depth is profile.depth
    assert result.timing["depth_total_ms"] == pytest.approx(profile.total_ms)


def test_macos_stream_native_coreml_defers_gpu_pack_to_output(monkeypatch):
    class NativeFrame:
        source_size = (4, 2)
        depth_width = 2
        depth_height = 2
        finite_depth = True
        nonfinite_count = 0
        preprocess_ms = 0.5
        model_ms = 4.0
        postprocess_ms = 0.2
        input_shared = True
        output_backing_used = True
        output_zero_copy = False
        released = False

        def configure_warp(self, **values):
            self.warp_values = values

        def pack(self, destination, output_size, output_format, *, rgb=False):
            assert output_size == (4, 2)
            assert output_format == "half_tab"
            assert rgb is True
            for offset in range(0, len(destination), 3):
                destination[offset : offset + 3] = bytes((10, 20, 30))

        def release(self):
            self.released = True

    class NativeProvider(FakeDepthProvider):
        def __init__(self):
            super().__init__()
            self.info = DepthProviderInfo(
                provider="fake-coreml",
                model_name="fake-coreml",
                model_id="xingyang1/Distill-Any-Depth-Small-hf",
                depth_resolution=2,
                cache_dir=".",
                depth_backend="coreml",
                runtime="coreml",
            )
            self.native_frame = NativeFrame()

        def predict_profile(self, rgb):
            raise AssertionError("native stream must not use Python depth path")

        def predict_profile_native(self, rgb, pixel_buffer, frame_id):
            assert pixel_buffer is not None
            assert frame_id == 7
            return DepthProfileResult(
                depth=self.native_frame,
                preprocess_ms=0.5,
                model_ms=4.0,
                postprocess_ms=0.2,
                finite_depth=True,
                native_depth=self.native_frame,
            )

    monkeypatch.setattr(runtime_module.sys, "platform", "darwin")
    monkeypatch.setenv("D2S_MAC_STREAM_NATIVE_IO", "1")
    provider = NativeProvider()
    runtime = StereoRuntime(
        StereoRuntimeConfig(
            model_id="xingyang1/Distill-Any-Depth-Small-hf",
            device="cpu",
            depth_backend="pytorch_mps",
            stereo_quality="quality_4k",
            output_format="half_tab",
            temporal=False,
            hole_fill="none",
        ),
        depth_provider=provider,
        collect_memory_stats=False,
    )

    result = runtime.process_rgb_frame(
        torch.empty(1, 3, 2, 4),
        pixel_buffer=object(),
        capture_frame_id=7,
    )

    assert result.sbs.shape == (1, 3, 2, 4)
    assert result.native_stream_frame is provider.native_frame
    assert result.native_stream_fallback is not None
    assert result.output_pack_backend == "native_coreml_metal_stream_pack"
    assert result.debug_info["native_coreml_stream_pack"] == 0
    assert result.debug_info["native_coreml_stream_pack_deferred"] == 1
    assert result.debug_info["native_coreml_input_shared"] == 1
    assert result.depth_complete is True
    assert provider.native_frame.released is False
    runtime.close()


def test_runtime_inference_gate_pauses_processing_and_reports_state():
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=provider, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 8, 12)

    runtime.set_inference_active(False)
    with pytest.raises(RuntimeError, match="inference is paused"):
        runtime.process_rgb_frame(rgb)
    assert runtime.to_report()["inference_active"] is False

    runtime.set_inference_active(True)
    runtime.process_rgb_frame(rgb)
    assert provider.predict_count == 1


def test_runtime_dynamic_convergence_uses_depth_quantile_in_debug_info():
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        temporal=False,
        max_disparity_px=18.0,
        convergence=0.0,
        dynamic_convergence_enabled=True,
        dynamic_convergence_strength=1.0,
        dynamic_convergence_target=1.0,
        dynamic_convergence_alpha=0.0,
        depth_antialias_strength=0.0,
    )
    runtime = StereoRuntime(config, depth_provider=provider, stats_window=4, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 8, 12)

    result = runtime.process_rgb_frame(rgb)

    assert result.debug_info["dynamic_convergence_enabled"] is True
    assert result.debug_info["dynamic_convergence_measured"] == pytest.approx(1.0)
    assert result.debug_info["dynamic_convergence_effective"] == pytest.approx(1.0)
    assert result.debug_info["convergence"] == pytest.approx(1.0)


def test_download_progress_prints_structured_event(capsys):
    progress = DownloadProgress(total=100, desc="model.safetensors", mininterval=0)
    progress.update(50)
    progress.update(50)
    progress.close()

    out = capsys.readouterr().out
    assert "[D2S_PROGRESS]" in out
    assert '"desc":"model.safetensors"' in out
    assert '"percent":100.0' in out


def test_download_progress_reports_download_size(capsys):
    progress = DownloadProgress(total=100, desc="model.safetensors", mininterval=0)
    progress.update(10)
    progress.update(40)
    progress.close()

    out = capsys.readouterr().out
    assert '"downloaded":"50 B"' in out
    assert '"size":"100 B"' in out


def test_hf_endpoint_candidates_default_to_mirror_then_official(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(
        "stereo_runtime.depth_provider.huggingface_endpoint_candidates",
        lambda: ("https://hf-mirror.com", "https://huggingface.co"),
    )

    assert _hf_endpoint_candidates() == ("https://hf-mirror.com", "https://huggingface.co")

    monkeypatch.setenv("HF_ENDPOINT", "https://custom.example")

    assert _hf_endpoint_candidates() == ("https://custom.example",)


def test_hf_endpoint_candidates_keep_default_fallback_when_env_is_auto_endpoint(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    monkeypatch.setattr(
        "stereo_runtime.depth_provider.huggingface_endpoint_candidates",
        lambda: ("https://huggingface.co", "https://hf-mirror.com"),
    )

    assert _hf_endpoint_candidates() == ("https://huggingface.co", "https://hf-mirror.com")


def test_hf_resolve_url_formats_download_links():
    assert _hf_resolve_url("https://hf-mirror.com/", "test/model") == "https://hf-mirror.com/test/model"
    assert _hf_resolve_url("https://hf-mirror.com/", "test/model", "model.safetensors") == "https://hf-mirror.com/test/model/resolve/main/model.safetensors"


class FakeProbeResponse:
    status = 206
    url = "https://cdn.example/model.safetensors"
    headers = {"Content-Length": "1", "Content-Type": "application/octet-stream"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status


def test_probe_download_url_logs_http_status(capsys):
    requests = []

    def opener(request, timeout):
        requests.append((request.get_method(), request.headers.get("Range"), timeout))
        return FakeProbeResponse()

    _probe_download_url("https://hf-mirror.com/test/model/resolve/main/model.safetensors", opener=opener)

    out = capsys.readouterr().out
    assert requests == [("HEAD", None, 10)]
    assert "Download probe HEAD: HTTP 206" in out
    assert "final=https://cdn.example/model.safetensors" in out


def test_probe_download_url_falls_back_to_range_get(capsys):
    from urllib.error import URLError

    requests = []

    def opener(request, timeout):
        requests.append((request.get_method(), request.headers.get("Range"), timeout))
        if len(requests) == 1:
            raise URLError("HEAD blocked")
        return FakeProbeResponse()

    _probe_download_url("https://hf-mirror.com/test/model/resolve/main/model.safetensors", opener=opener)

    out = capsys.readouterr().out
    assert requests == [("HEAD", None, 10), ("GET", "bytes=0-0", 10)]
    assert "Download probe HEAD failed: URLError" in out
    assert "Download probe GET: HTTP 206" in out


def test_reachable_hf_endpoints_raises_when_all_probes_fail(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr("stereo_runtime.depth_provider._probe_download_url", lambda url: False)

    with pytest.raises(RuntimeError, match="check your network or enable VPN"):
        _reachable_hf_endpoints("test/model")


def test_resolve_hf_model_file_reuses_cached_path(monkeypatch, tmp_path, capsys):
    import sys
    import types

    _RESOLVED_HF_MODEL_FILES.clear()
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"weights")
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(model_file)

    fake_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    try:
        first = _resolve_hf_model_file("test/model", tmp_path)
        second = _resolve_hf_model_file("test/model", tmp_path)

        out = capsys.readouterr().out
        assert first == second == str(model_file)
        assert len(calls) == 1
        assert out.count("Depth model cache hit:") == 1
    finally:
        _RESOLVED_HF_MODEL_FILES.clear()


def test_resolve_hf_model_file_falls_back_when_hf_returns_empty_file(monkeypatch, tmp_path):
    import sys
    import types

    _RESOLVED_HF_MODEL_FILES.clear()
    empty_file = tmp_path / "empty.safetensors"
    empty_file.write_bytes(b"")
    direct_file = tmp_path / "models--test--model" / "model.safetensors"
    calls = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        empty_file.write_bytes(b"")
        return str(empty_file)

    def fake_direct(model_id, filename, cache_dir, endpoint):
        direct_file.parent.mkdir(parents=True, exist_ok=True)
        direct_file.write_bytes(b"weights")
        return str(direct_file)

    fake_module = types.SimpleNamespace(hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr("stereo_runtime.depth_provider._reachable_hf_endpoints", lambda model_id: ("https://hf-mirror.com",))
    monkeypatch.setattr("stereo_runtime.depth_provider._probe_download_url", lambda url: True)
    monkeypatch.setattr("stereo_runtime.depth_provider._download_hf_file_direct", fake_direct)

    try:
        assert _resolve_hf_model_file("test/model", tmp_path) == str(direct_file)
        assert len(calls) >= 2
        assert calls[-1]["force_download"] is True
    finally:
        _RESOLVED_HF_MODEL_FILES.clear()


def test_hf_endpoint_fallback_restores_environment(monkeypatch, capsys):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr("stereo_runtime.depth_provider._reachable_hf_endpoints", lambda model_id: ("https://hf-mirror.com", "https://huggingface.co"))
    monkeypatch.setattr("stereo_runtime.depth_provider._probe_download_url", lambda url: True)
    calls = []

    def load(model_id):
        calls.append((model_id, os.environ.get("HF_ENDPOINT")))
        if len(calls) == 1:
            raise RuntimeError("mirror down")
        return "model"

    assert _load_hf_with_endpoint_fallback(load, "test/model") == "model"
    assert calls == [
        ("test/model", "https://hf-mirror.com"),
        ("test/model", "https://huggingface.co"),
    ]
    out = capsys.readouterr().out
    assert "Model download URL: https://hf-mirror.com/test/model" in out
    assert "Model download URL: https://huggingface.co/test/model" in out
    assert "HF_ENDPOINT" not in os.environ


def test_hf_download_progress_patch_is_scoped_to_context():
    import importlib

    file_download = importlib.import_module("huggingface_hub.file_download")
    hf_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")
    snapshot_download = importlib.import_module("huggingface_hub._snapshot_download")

    original_file_download_tqdm = file_download.tqdm
    original_hf_tqdm = hf_tqdm.tqdm
    original_snapshot_tqdm = snapshot_download.hf_tqdm

    with _hf_download_progress_patch():
        assert file_download.tqdm is DownloadProgress
        assert hf_tqdm.tqdm is DownloadProgress
        assert snapshot_download.hf_tqdm is DownloadProgress

    assert file_download.tqdm is original_file_download_tqdm
    assert hf_tqdm.tqdm is original_hf_tqdm
    assert snapshot_download.hf_tqdm is original_snapshot_tqdm


def test_download_preparing_progress_prints_waiting_status_without_fake_bar(capsys):
    _print_download_preparing_progress("model.safetensors")

    out = capsys.readouterr().out
    assert "[Main] Preparing download model.safetensors: waiting for server response..." in out
    assert "0.00%" not in out
    assert "[>" not in out


def test_progress_print_uses_plain_logging(capsys):
    _progress_print("[Main] Runtime preparation: checking depth model test/model")

    assert "[Main] Runtime preparation: checking depth model test/model" in capsys.readouterr().out


def test_raise_model_resolution_error_includes_last_exception():
    try:
        _raise_model_resolution_error("lc700x/InfiniDepth-Large", ValueError("mirror timeout"), local_only=False)
    except RuntimeError as exc:
        text = str(exc)
        assert "unable to resolve InfiniDepth weights" in text
        assert "ValueError: mirror timeout" in text
    else:
        raise AssertionError("expected RuntimeError")


def test_incomplete_download_progress_close_does_not_redraw_last_progress(capsys):
    progress = DownloadProgress(total=100, desc="model.safetensors")
    progress.update(25)
    capsys.readouterr()

    progress.close()

    out = capsys.readouterr().out
    assert "Downloading model.safetensors" not in out


def test_runtime_slow_frame_log_is_debug_only(monkeypatch, caplog, capsys):
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=provider, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 8, 12)

    monkeypatch.setenv("D2S_SLOW_RUNTIME_LOG_MS", "0")
    with caplog.at_level(logging.DEBUG, logger="stereo_runtime.runtime"):
        runtime.process_rgb_frame(rgb)

    assert "[StereoRuntime] slow frame:" in caplog.text
    assert "[StereoRuntime] slow frame:" not in capsys.readouterr().out


def test_runtime_frame_refresh_debug_log_is_capped(monkeypatch, caplog):
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=provider, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 8, 12)

    monkeypatch.setenv("D2S_SLOW_RUNTIME_LOG_MS", "999999")
    monkeypatch.setenv("D2S_RUNTIME_FRAME_LOG_REFRESH_S", "0.001")
    with caplog.at_level(logging.DEBUG, logger="stereo_runtime.runtime"):
        for _ in range(7):
            runtime._last_runtime_perf_log_ts = -1e9
            runtime.process_rgb_frame(rgb)

    assert caplog.text.count("[StereoRuntime] frame refresh:") == 5

def test_rolling_runtime_stats_reports_percentiles_fps_and_memory():
    stats = RollingRuntimeStats(maxlen=3)
    stats.update({"total_ms": 10.0, "synthesis_ms": 4.0}, {"cuda_peak_memory_allocated_mb": 100.0})
    stats.update({"total_ms": 20.0, "synthesis_ms": 5.0}, {"cuda_peak_memory_allocated_mb": 150.0})
    stats.update({"total_ms": 30.0, "synthesis_ms": 6.0}, {"cuda_peak_memory_allocated_mb": 200.0})
    stats.update({"total_ms": 40.0, "synthesis_ms": 7.0}, {"cuda_peak_memory_allocated_mb": 250.0})

    report = stats.to_report()

    assert report["count"] == 3
    assert report["stages"]["total_ms"]["min"] == 20.0
    assert report["stages"]["total_ms"]["max"] == 40.0
    assert report["stages"]["total_ms"]["mean"] == 30.0
    assert report["stages"]["total_ms"]["median"] == 30.0
    assert report["stages"]["total_ms"]["p90"] == 38.0
    assert report["stages"]["total_ms"]["p99"] == pytest.approx(39.8)
    assert report["fps"]["latest"] == 25.0
    assert report["fps"]["mean_from_mean_ms"] == pytest.approx(1000.0 / 30.0)
    assert report["memory"]["cuda_peak_memory_allocated_mb"]["max"] == 250.0

    stats.reset()
    assert stats.to_report()["count"] == 0


def test_stereo_runtime_exports_new_public_names():
    from stereo_runtime import StereoRuntime, StereoRuntimeConfig
    from stereo_runtime.depth_provider import DepthProviderConfig
    from stereo_runtime.runtime import RollingRuntimeStats as AliasRollingStats

    assert StereoRuntime.__name__ == "StereoRuntime"
    assert StereoRuntimeConfig.__name__ == "StereoRuntimeConfig"
    assert DepthProviderConfig.__name__ == "DepthProviderConfig"
    assert AliasRollingStats is RollingRuntimeStats


def test_fast_plus_fused_uses_resolved_parallax_budget_contract():
    fused_source = (APP_ROOT / "stereo_runtime" / "fast_plus_fused_triton.py").read_text(encoding="utf-8")
    runtime_source = (APP_ROOT / "stereo_runtime" / "runtime.py").read_text(encoding="utf-8")

    assert "max_disparity_px: tl.constexpr" in fused_source
    assert "max_disparity_px: float" in fused_source
    assert "depth_strength: tl.constexpr" in fused_source
    assert "depth_strength: float" in fused_source
    assert "max_disparity_px * depth_strength * 0.5" in fused_source
    assert "effective_ipd_m" not in fused_source
    assert "max_shift_ratio" not in fused_source
    assert "width *" not in fused_source
    assert "_fast_plus_hole_ratio_probe_kernel" in fused_source
    assert "_fast_plus_active_mask_kernel" in fused_source
    assert "_fast_plus_sparse_fill_kernel" in fused_source
    assert "D2S_TRITON_SPARSE_HOLE_THRESHOLD" in fused_source
    assert "max_disparity_px=float(budget.max_disparity_px)" in runtime_source
    assert 'depth_strength=max(0.0, float(getattr(stereo_config, "depth_strength", 1.0)))' in runtime_source


def test_fast_plus_fused_runtime_is_bypassed_when_hole_fill_is_disabled(monkeypatch):
    import stereo_runtime.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_fast_plus_fused_enabled", lambda: True)
    runtime = StereoRuntime.__new__(StereoRuntime)
    result, reason = runtime._try_fast_plus_fused_sbs(
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 1, 8, 8),
        StereoConfig(backend="fast_plus", hole_fill="none", hole_fill_mode="none"),
    )

    assert result is None
    assert reason == "hole_fill_disabled"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fast_plus_fused Triton")
def test_fast_plus_fused_runtime_emits_uint8_half_sbs(monkeypatch):
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast_plus",
        output_format="half_sbs",
        temporal=True,
    )
    runtime = StereoRuntime(config, depth_provider=provider, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 16, 32, device="cuda", dtype=torch.float32)

    monkeypatch.setenv("D2S_RUNTIME_OUTPUT_UINT8", "1")
    monkeypatch.setenv("D2S_FAST_PLUS_FUSED", "1")
    result = runtime.process_rgb_frame(rgb)
    torch.cuda.synchronize()

    assert result.sbs.shape == rgb.shape
    assert result.sbs.dtype == torch.uint8
    assert result.debug_info["runtime_output_pack_backend"] in {
        "triton_half_sbs_uint8",
        "torch_packed_sbs_to_uint8",
    }
    assert result.debug_info["fast_plus_fused_backend"] in {
        "not_used",
        result.debug_info["runtime_output_pack_backend"],
    }
    expected_temporal_bypass = (
        0 if result.debug_info["fast_plus_fused_backend"] == "not_used" else 1
    )
    assert result.debug_info["fast_plus_fused_temporal_bypass"] == expected_temporal_bypass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for direct SBS")
def test_quality_direct_sbs_uint8_preserves_packed_stereo_views(monkeypatch):
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="quality_4k",
        output_format="half_sbs",
        temporal=False,
        depth_strength=1.0,
        max_disparity_px=12.0,
    )
    runtime = StereoRuntime(
        config,
        depth_provider=provider,
        collect_memory_stats=False,
    )
    runtime.configure_stereo(
        StereoConfig(
            backend="quality_4k",
            layers=2,
            output_format="half_sbs",
            temporal=False,
            hole_fill="none",
            hole_fill_mode="none",
            depth_strength=1.0,
            max_disparity_px=12.0,
        )
    )
    rgb = torch.rand(1, 3, 24, 40, device="cuda", dtype=torch.float32)

    monkeypatch.delenv("D2S_RUNTIME_OUTPUT_UINT8", raising=False)
    expected_float = runtime.process_rgb_frame(rgb).sbs
    monkeypatch.setenv("D2S_RUNTIME_OUTPUT_UINT8", "1")
    actual = runtime.process_rgb_frame(rgb)
    torch.cuda.synchronize()

    expected = expected_float.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    assert torch.equal(actual.sbs, expected)
    half_width = actual.sbs.shape[-1] // 2
    assert not torch.equal(
        actual.sbs[..., :half_width],
        actual.sbs[..., half_width:],
    )
    assert actual.debug_info["runtime_output_pack_backend"] == "triton_warp_composite2_half_sbs_uint8"
