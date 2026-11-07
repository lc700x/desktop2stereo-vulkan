from pathlib import Path
from types import SimpleNamespace
import os

import pytest

from stereo_runtime import (
    artifact_paths_for_model,
    prepare_model_artifacts,
)
from stereo_runtime.model_artifacts import (
    ensure_model_downloaded,
    ensure_onnx_exported,
    find_local_model_snapshot,
    find_local_model_weight,
    select_existing_migraphx,
    select_existing_onnx,
    select_existing_trt,
)
from stereo_runtime.model_registry import ModelRegistry


def test_find_local_model_snapshot_resolves_hub_revision_without_refs(
    tmp_path: Path,
):
    spec = ModelRegistry.default().get("Distill-Any-Depth-Large")
    snapshot = spec.model_dir(tmp_path) / "snapshots" / "0123456789abcdef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "pytorch_model.bin").write_bytes(b"cached weights")

    assert find_local_model_snapshot(spec.model_id, tmp_path) == snapshot


def test_find_local_model_snapshot_prefers_main_ref(tmp_path: Path):
    spec = ModelRegistry.default().get("Distill-Any-Depth-Large")
    model_dir = spec.model_dir(tmp_path)
    main_snapshot = model_dir / "snapshots" / "main-revision"
    newer_snapshot = model_dir / "snapshots" / "newer-revision"
    for snapshot in (main_snapshot, newer_snapshot):
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "pytorch_model.bin").write_bytes(b"cached weights")
    refs = model_dir / "refs"
    refs.mkdir()
    (refs / "main").write_text("main-revision\n", encoding="utf-8")

    assert find_local_model_snapshot(spec.model_id, tmp_path) == main_snapshot


def test_artifact_paths_follow_d2s_naming():
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir="models", export_height=294, export_width=518)

    assert paths.model_id == "lc700x/Distill-Any-Depth-Base-hf"
    assert str(paths.model_dir).replace("\\", "/") == "models/models--lc700x--Distill-Any-Depth-Base-hf"
    assert paths.onnx_fp16_path.name == "model_fp16_294x518.onnx"
    assert paths.onnx_fp32_path.name == "model_fp32_294x518.onnx"
    assert paths.trt_fp16_path.name == "model_fp16_294x518.trt"
    assert paths.migraphx_fp16_path.name == "model_fp16_294x518.mgx"


def test_infinidepth_artifact_paths_use_patch_16_export_size():
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir="models", export_height=294, export_width=518)

    assert paths.model_id == "lc700x/InfiniDepth-Base"
    assert paths.onnx_fp16_path.name == "model_fp16_288x512.onnx"
    assert paths.onnx_fp32_path.name == "model_fp32_288x512.onnx"
    assert paths.trt_fp16_path.name == "model_fp16_288x512.trt"


def test_video_model_cache_accepts_named_pth_checkpoint(tmp_path: Path):
    spec = ModelRegistry.default().get("Video-Depth-Anything-Small")
    model_dir = spec.model_dir(tmp_path) / "snapshots" / "local"
    model_dir.mkdir(parents=True)
    (model_dir / "video_depth_anything_vits.pth").write_bytes(b"weights")

    assert find_local_model_weight(spec.model_dir(tmp_path)).name == "video_depth_anything_vits.pth"
    assert ensure_model_downloaded(spec, cache_dir=tmp_path, local_files_only=True) == spec.model_dir(tmp_path)


def test_onnx_export_uses_patch_aligned_artifact_size(monkeypatch, tmp_path: Path):
    spec = ModelRegistry.default().get("Video-Depth-Anything-Small")
    model_dir = spec.model_dir(tmp_path)
    model_dir.mkdir(parents=True)
    (model_dir / "video_depth_anything_vits.pth").write_bytes(b"weights")
    calls = {}

    def fake_export(**kwargs):
        calls.update(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"onnx")
        return SimpleNamespace(output_path=output_path)

    monkeypatch.setattr("stereo_runtime.onnx_export.export_depth_model_onnx", fake_export)
    ensure_onnx_exported(
        spec,
        cache_dir=tmp_path,
        height=191,
        width=336,
        dtype="fp16",
        local_files_only=True,
        export_if_missing=True,
    )

    assert (calls["height"], calls["width"]) == (196, 336)


def test_select_existing_onnx_prefers_fp16_for_auto(tmp_path: Path):
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.onnx_fp16_path.write_bytes(b"fp16")
    paths.onnx_fp32_path.write_bytes(b"fp32")

    assert select_existing_onnx(paths, "auto") == paths.onnx_fp16_path
    assert select_existing_onnx(paths, "fp32") == paths.onnx_fp32_path


def test_select_existing_onnx_reuses_fp32_fallback_for_fp16_request(tmp_path: Path):
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.onnx_fp32_path.write_bytes(b"fp32")

    assert select_existing_onnx(paths, "fp16") == paths.onnx_fp32_path


def test_select_existing_trt_reuses_fp32_fallback_for_fp16_request(tmp_path: Path):
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    fp32_trt = paths.trt_path_for_dtype("fp32")
    fp32_trt.write_bytes(b"trt")

    assert select_existing_trt(paths, "fp16") == fp32_trt


def test_find_local_model_weight_prefers_direct_weight_file(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"weights")

    assert find_local_model_weight(model_dir) == weight


def test_find_local_model_weight_ignores_empty_weight_file(tmp_path: Path):
    model_dir = tmp_path / "model"
    nested = model_dir / "snapshots" / "ok"
    nested.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"")
    weight = nested / "model.safetensors"
    weight.write_bytes(b"weights")

    assert find_local_model_weight(model_dir) == weight


def test_prepare_model_artifacts_reports_missing_without_side_effects(tmp_path: Path):
    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        local_files_only=False,
        download_if_missing=False,
        export_onnx_if_missing=False,
        build_trt_if_missing=False,
    )

    assert result.downloaded is False
    assert result.onnx_ready is False
    assert result.trt_ready is False
    assert "onnx artifact missing" in result.notes


def test_prepare_model_artifacts_local_files_only_requires_model_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        prepare_model_artifacts("Distill-Any-Depth-Base", cache_dir=tmp_path, local_files_only=True)


def test_prepare_model_artifacts_passes_local_files_only_to_onnx_export(monkeypatch, tmp_path: Path):
    calls = {}
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)

    def fake_export(**kwargs):
        calls.update(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"onnx")
        return SimpleNamespace(output_path=output_path)

    import stereo_runtime.onnx_export as onnx_export

    def fake_snapshot_download(**kwargs):
        (paths.model_dir / "model.safetensors").write_bytes(b"weights")
        (paths.model_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(paths.model_dir / "snapshots" / "fake")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("stereo_runtime.depth_provider._reachable_hf_endpoints", lambda model_id: ("https://hf-mirror.com",))
    monkeypatch.setattr(onnx_export, "export_depth_model_onnx", fake_export)

    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        local_files_only=True,
        export_onnx_if_missing=True,
    )

    assert result.onnx_ready is True
    assert calls["local_files_only"] is True
    assert calls["force_download"] is False


def test_prepare_model_artifacts_reuses_fp32_fallback_for_fp16_request(monkeypatch, tmp_path: Path):
    import stereo_runtime.onnx_export as onnx_export

    def fail_export(**kwargs):
        raise AssertionError("ONNX export should not run when fp32 fallback already exists")

    monkeypatch.setattr(onnx_export, "export_depth_model_onnx", fail_export)
    monkeypatch.setattr("stereo_runtime.model_artifacts.ensure_model_downloaded", lambda *args, **kwargs: None)
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.onnx_fp32_path.write_bytes(b"onnx")

    result = prepare_model_artifacts(
        "InfiniDepth-Base",
        cache_dir=tmp_path,
        onnx_dtype="fp16",
        export_onnx_if_missing=True,
    )

    assert result.onnx_ready is True
    assert result.selected_onnx_path == paths.onnx_fp32_path


def test_prepare_model_artifacts_skips_download_when_local_onnx_exists(monkeypatch, tmp_path: Path):
    def fail_download(*args, **kwargs):
        raise AssertionError("model download should not run when ONNX already exists")

    monkeypatch.setattr("stereo_runtime.model_artifacts.ensure_model_downloaded", fail_download)
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.onnx_fp32_path.write_bytes(b"onnx")

    result = prepare_model_artifacts(
        "InfiniDepth-Base",
        cache_dir=tmp_path,
        onnx_dtype="fp16",
        download_if_missing=True,
        export_onnx_if_missing=True,
    )

    assert result.selected_onnx_path == paths.onnx_fp32_path


def test_prepare_model_artifacts_skips_download_and_export_when_local_trt_exists(monkeypatch, tmp_path: Path):
    def fail_download(*args, **kwargs):
        raise AssertionError("model download should not run when TRT already exists")

    monkeypatch.setattr("stereo_runtime.model_artifacts.ensure_model_downloaded", fail_download)
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    trt_path = paths.trt_path_for_dtype("fp32")
    trt_path.write_bytes(b"trt")

    result = prepare_model_artifacts(
        "InfiniDepth-Base",
        cache_dir=tmp_path,
        onnx_dtype="fp16",
        download_if_missing=True,
        export_onnx_if_missing=True,
        artifact_backend="tensorrt",
        build_trt_if_missing=True,
    )

    assert result.trt_ready is True
    assert result.selected_onnx_path is None


def test_prepare_model_artifacts_does_not_use_trt_for_onnx_backend(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_export(**kwargs):
        calls.update(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"onnx")
        return SimpleNamespace(output_path=output_path)

    import stereo_runtime.onnx_export as onnx_export

    monkeypatch.setattr("stereo_runtime.model_artifacts.ensure_model_downloaded", lambda *args, **kwargs: None)
    monkeypatch.setattr(onnx_export, "export_depth_model_onnx", fake_export)
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.trt_path_for_dtype("fp32").write_bytes(b"trt")

    result = prepare_model_artifacts(
        "InfiniDepth-Base",
        cache_dir=tmp_path,
        onnx_dtype="fp32",
        artifact_backend="onnx",
        export_onnx_if_missing=True,
    )

    assert result.selected_onnx_path == paths.onnx_fp32_path
    assert result.trt_ready is False


def test_prepare_model_artifacts_uses_migraphx_only_for_migraphx_backend(monkeypatch, tmp_path: Path):
    def fail_download(*args, **kwargs):
        raise AssertionError("model download should not run when MIGraphX graph already exists")

    monkeypatch.setattr("stereo_runtime.model_artifacts.ensure_model_downloaded", fail_download)
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.migraphx_fp16_path.write_bytes(b"mgx")

    assert select_existing_migraphx(paths, "auto") == paths.migraphx_fp16_path

    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        artifact_backend="migraphx",
        export_onnx_if_missing=True,
        build_migraphx_if_missing=True,
    )

    assert result.migraphx_ready is True
    assert result.selected_migraphx_path == paths.migraphx_fp16_path
    assert result.selected_onnx_path is None


def test_prepare_model_artifacts_uses_local_weight_before_download(monkeypatch, tmp_path: Path):
    calls = {}

    def fail_download(*args, **kwargs):
        raise AssertionError("model download should not run when local weight exists")

    def fake_export(**kwargs):
        calls.update(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"onnx")
        return SimpleNamespace(output_path=output_path)

    import stereo_runtime.onnx_export as onnx_export

    monkeypatch.setattr("stereo_runtime.model_artifacts.ensure_model_downloaded", fail_download)
    monkeypatch.setattr(onnx_export, "export_depth_model_onnx", fake_export)
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    (paths.model_dir / "model.safetensors").write_bytes(b"weights")

    result = prepare_model_artifacts(
        "InfiniDepth-Base",
        cache_dir=tmp_path,
        onnx_dtype="fp32",
        download_if_missing=True,
        export_onnx_if_missing=True,
    )

    assert result.selected_onnx_path == paths.onnx_fp32_path
    assert calls["local_files_only"] is True


def test_prepare_model_artifacts_confirms_model_before_onnx_export(monkeypatch, tmp_path: Path):
    calls = []
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)

    def fake_snapshot_download(**kwargs):
        calls.append("download")
        paths.model_dir.mkdir(parents=True, exist_ok=True)
        (paths.model_dir / "model.safetensors").write_bytes(b"weights")
        (paths.model_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(paths.model_dir / "snapshots" / "fake")

    def fake_export(**kwargs):
        calls.append("onnx")
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"onnx")
        return SimpleNamespace(output_path=output_path)

    import stereo_runtime.onnx_export as onnx_export

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("stereo_runtime.depth_provider._reachable_hf_endpoints", lambda model_id: ("https://hf-mirror.com",))
    monkeypatch.setattr(onnx_export, "export_depth_model_onnx", fake_export)

    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        local_files_only=False,
        download_if_missing=False,
        export_onnx_if_missing=True,
    )

    assert result.onnx_ready is True
    assert calls == ["download", "onnx"]


def test_infinidepth_download_requires_resolved_weight_file(monkeypatch, tmp_path: Path):
    calls = {}
    spec = ModelRegistry.default().get("InfiniDepth-Base")
    model_dir = spec.model_dir(tmp_path)
    model_dir.mkdir(parents=True)

    def fake_resolve(model_id, cache_dir, *, local_files_only=False, force_download=False):
        calls["model_id"] = model_id
        calls["cache_dir"] = Path(cache_dir)
        calls["local_files_only"] = local_files_only
        calls["force_download"] = force_download
        return str(model_dir / "model.safetensors")

    import stereo_runtime.depth_provider as depth_provider

    monkeypatch.setattr(depth_provider, "_resolve_hf_model_file", fake_resolve)

    assert ensure_model_downloaded(spec, cache_dir=tmp_path, local_files_only=False) == model_dir
    assert calls == {
        "model_id": "lc700x/InfiniDepth-Base",
        "cache_dir": tmp_path,
        "local_files_only": False,
        "force_download": False,
    }


@pytest.mark.parametrize(
    "model_name",
    ["DA3-BASE", "Video-Depth-Anything-Base", "Metric-Video-Depth-Anything-Base"],
)
def test_local_config_model_families_do_not_require_remote_config(
    model_name: str, tmp_path: Path
):
    spec = ModelRegistry.default().get(model_name)
    model_dir = spec.model_dir(tmp_path)
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")

    from stereo_runtime.model_artifacts import _model_cache_is_complete

    assert _model_cache_is_complete(spec, model_dir) is True


def test_generic_download_confirms_snapshot_even_when_cache_dir_exists(monkeypatch, tmp_path: Path):
    calls = {}
    spec = ModelRegistry.default().get("Distill-Any-Depth-Base")
    model_dir = spec.model_dir(tmp_path)
    model_dir.mkdir(parents=True)

    def fake_snapshot_download(**kwargs):
        (model_dir / "model.safetensors").write_bytes(b"weights")
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        calls.update(kwargs)
        return str(model_dir / "snapshots" / "fake")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("stereo_runtime.depth_provider._reachable_hf_endpoints", lambda model_id: ("https://hf-mirror.com",))

    assert ensure_model_downloaded(spec, cache_dir=tmp_path, local_files_only=False) == model_dir
    assert calls["repo_id"] == "lc700x/Distill-Any-Depth-Base-hf"
    assert calls["cache_dir"] == str(tmp_path)
    assert calls["local_files_only"] is False
    assert calls["force_download"] is False


def test_generic_snapshot_download_uses_selected_hf_endpoint(monkeypatch, tmp_path: Path):
    spec = ModelRegistry.default().get("Distill-Any-Depth-Base")
    model_dir = spec.model_dir(tmp_path)
    observed = []

    def fake_snapshot_download(**kwargs):
        observed.append(os.environ.get("HF_ENDPOINT"))
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.safetensors").write_bytes(b"weights")
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(model_dir / "snapshots" / "fake")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        "stereo_runtime.depth_provider._reachable_hf_endpoints",
        lambda model_id: ("https://hf-mirror.com",),
    )

    assert ensure_model_downloaded(spec, cache_dir=tmp_path) == model_dir
    assert observed == ["https://hf-mirror.com"]


def test_generic_snapshot_download_uses_structured_progress_patch(monkeypatch, tmp_path: Path):
    from contextlib import contextmanager

    events = []
    spec = ModelRegistry.default().get("Distill-Any-Depth-Base")
    model_dir = spec.model_dir(tmp_path)
    model_dir.mkdir(parents=True)

    @contextmanager
    def fake_progress_patch():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def fake_snapshot_download(**kwargs):
        events.append("download")
        (model_dir / "model.safetensors").write_bytes(b"weights")
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(model_dir / "snapshots" / "fake")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("stereo_runtime.depth_provider._hf_download_progress_patch", fake_progress_patch)
    monkeypatch.setattr("stereo_runtime.depth_provider._reachable_hf_endpoints", lambda model_id: ("https://hf-mirror.com",))

    assert ensure_model_downloaded(spec, cache_dir=tmp_path, local_files_only=False) == model_dir
    assert events == ["enter", "download", "exit"]


def test_generic_snapshot_download_falls_back_when_weight_is_empty(monkeypatch, tmp_path: Path):
    spec = ModelRegistry.default().get("Distill-Any-Depth-Base")
    model_dir = spec.model_dir(tmp_path)
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"")
    direct_calls = []

    def fake_snapshot_download(**kwargs):
        return str(model_dir / "snapshots" / "fake")

    def fake_direct(model_id, filename, cache_dir, endpoint):
        direct_calls.append((model_id, filename, endpoint))
        target = model_dir / filename
        target.write_bytes(b"weights")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("stereo_runtime.depth_provider._reachable_hf_endpoints", lambda model_id: ("https://hf-mirror.com",))
    monkeypatch.setattr("stereo_runtime.depth_provider._download_hf_file_direct", fake_direct)

    assert ensure_model_downloaded(spec, cache_dir=tmp_path, local_files_only=False) == model_dir
    assert direct_calls == [
        ("lc700x/Distill-Any-Depth-Base-hf", "model.safetensors", "https://hf-mirror.com"),
        ("lc700x/Distill-Any-Depth-Base-hf", "config.json", "https://hf-mirror.com"),
    ]


def test_generic_snapshot_download_error_falls_back_to_direct(monkeypatch, tmp_path: Path):
    spec = ModelRegistry.default().get("Distill-Any-Depth-Base")
    model_dir = spec.model_dir(tmp_path)
    direct_calls = []

    def fake_snapshot_download(**kwargs):
        raise RuntimeError("hub cache failed")

    def fake_direct(model_id, filename, cache_dir, endpoint):
        direct_calls.append((model_id, filename, endpoint))
        target = model_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"weights")
        return str(target)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("stereo_runtime.depth_provider._reachable_hf_endpoints", lambda model_id: ("https://hf-mirror.com",))
    monkeypatch.setattr("stereo_runtime.depth_provider._download_hf_file_direct", fake_direct)

    assert ensure_model_downloaded(spec, cache_dir=tmp_path, local_files_only=False) == model_dir
    assert direct_calls == [
        ("lc700x/Distill-Any-Depth-Base-hf", "model.safetensors", "https://hf-mirror.com"),
        ("lc700x/Distill-Any-Depth-Base-hf", "config.json", "https://hf-mirror.com"),
    ]


def test_prepare_model_artifacts_builds_trt_from_existing_onnx(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_build(onnx_path, engine_path, *, fp16=True, workspace_gb=4, force=False):
        calls["onnx_path"] = Path(onnx_path)
        calls["engine_path"] = Path(engine_path)
        calls["fp16"] = fp16
        calls["workspace_gb"] = workspace_gb
        calls["force"] = force
        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        Path(engine_path).write_bytes(b"trt")
        return Path(engine_path)

    import stereo_runtime.providers.nvidia.tensorrt_native as native_provider

    monkeypatch.setattr(native_provider, "build_native_tensorrt_engine", fake_build)
    paths = artifact_paths_for_model("Distill-Any-Depth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.onnx_fp16_path.write_bytes(b"onnx")

    result = prepare_model_artifacts(
        "Distill-Any-Depth-Base",
        cache_dir=tmp_path,
        artifact_backend="tensorrt",
        build_trt_if_missing=True,
        trt_workspace_gb=7,
    )

    assert result.trt_ready is True
    assert calls["onnx_path"] == paths.onnx_fp16_path
    assert calls["engine_path"] == paths.trt_fp16_path
    assert calls["fp16"] is True
    assert calls["workspace_gb"] == 7
    assert calls["force"] is False


def test_prepare_model_artifacts_builds_fp32_trt_from_fp32_onnx(monkeypatch, tmp_path: Path):
    calls = {}

    def fake_build(onnx_path, engine_path, *, fp16=True, workspace_gb=4, force=False):
        calls["onnx_path"] = Path(onnx_path)
        calls["engine_path"] = Path(engine_path)
        calls["fp16"] = fp16
        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        Path(engine_path).write_bytes(b"trt")
        return Path(engine_path)

    import stereo_runtime.providers.nvidia.tensorrt_native as native_provider

    monkeypatch.setattr(native_provider, "build_native_tensorrt_engine", fake_build)
    paths = artifact_paths_for_model("InfiniDepth-Base", cache_dir=tmp_path)
    paths.model_dir.mkdir(parents=True)
    paths.onnx_fp32_path.write_bytes(b"onnx")

    result = prepare_model_artifacts(
        "InfiniDepth-Base",
        cache_dir=tmp_path,
        onnx_dtype="fp32",
        artifact_backend="tensorrt",
        build_trt_if_missing=True,
    )

    assert result.trt_ready is True
    assert calls["onnx_path"] == paths.onnx_fp32_path
    assert calls["engine_path"] == paths.trt_path_for_dtype("fp32")
    assert calls["engine_path"].name == "model_fp32_288x512.trt"
    assert calls["fp16"] is False
