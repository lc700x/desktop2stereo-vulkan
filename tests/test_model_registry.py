import re
from pathlib import Path

from path_config import APP_ROOT

from gui.model_catalog import GUI_MODEL_CATALOG, V25_GUI_MODEL_NAMES
from stereo_runtime import DepthRuntimeConfig, ModelRegistry, resolve_model_dir
from stereo_runtime.adapter import _depth_export_size_from_settings, depth_provider_config_from_runtime
from stereo_runtime.model_artifacts import (
    ZIPDEPTH_CHECKPOINT_FILENAMES,
    _model_cache_is_complete,
    artifact_paths_for_model,
    ensure_model_downloaded,
)
from stereo_runtime.model_capabilities import DISABLE_COREML_KEYWORDS, DISABLE_MIGRAPHX_KEYWORDS

ROOT = Path(__file__).resolve().parents[1]


def test_default_registry_contains_d2s_models():
    registry = ModelRegistry.default()

    assert registry.resolve_model_id("Distill-Any-Depth-Base") == "lc700x/Distill-Any-Depth-Base-hf"
    assert registry.resolve_model_id("DepthPro-Large") == "apple/DepthPro-hf"
    assert registry.resolve_model_id("depth-anything/DA3-BASE") == "depth-anything/DA3-BASE"
    assert registry.resolve_model_id("depth-anything-indoor-large") == "lc700x/depth-anything-indoor-large-hf"
    assert registry.resolve_model_id("depth-anything-outdoor-large") == "lc700x/depth-anything-outdoor-large-hf"
    assert len(registry.names()) >= 42


def test_depth_runtime_config_resolves_model_dir_from_model_name():
    config = DepthRuntimeConfig(model_id="Distill-Any-Depth-Base", cache_dir="./models")

    assert config.resolved_model_id == "lc700x/Distill-Any-Depth-Base-hf"
    assert str(config.model_path).replace("\\", "/").endswith("models/models--lc700x--Distill-Any-Depth-Base-hf")
    assert config.onnx_path.name == "model_fp16_294x518.onnx"
    assert config.fp32_onnx_path.name == "model_fp32_294x518.onnx"
    assert config.trt_engine_path.name == "model_fp16_294x518.trt"


def test_resolve_model_dir_uses_huggingface_cache_name():
    path = resolve_model_dir("owner/repo-name", "cache-root")

    assert str(path).replace("\\", "/") == "cache-root/models--owner--repo-name"


def test_da3_registry_uses_correct_huggingface_repo_revisions():
    registry = ModelRegistry.default()

    assert registry.resolve_model_id("DA3-SMALL") == "depth-anything/DA3-SMALL"
    assert registry.resolve_model_id("DA3-BASE") == "depth-anything/DA3-BASE"
    assert registry.resolve_model_id("DA3-LARGE") == "depth-anything/DA3-LARGE-1.1"
    assert registry.resolve_model_id("DA3-GIANT") == "depth-anything/DA3-GIANT-1.1"
    assert registry.resolve_model_id("DA3NESTED-GIANT-LARGE") == "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    assert registry.resolve_model_id("DA3MONO-LARGE") == "depth-anything/DA3MONO-LARGE"


def test_da3_model_dir_uses_resolved_repo_id():
    config = DepthRuntimeConfig(model_id="DA3-LARGE", cache_dir="./models")

    assert config.resolved_model_id == "depth-anything/DA3-LARGE-1.1"
    assert str(config.model_path).replace("\\", "/").endswith("models/models--depth-anything--DA3-LARGE-1.1")


def test_depth_runtime_config_passes_trt_build_options():
    config = DepthRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        depth_backend="tensorrt_native",
        build_trt_engine=True,
        force_rebuild_trt=True,
        use_cuda_graph=True,
    )

    provider_config = depth_provider_config_from_runtime(config)

    assert provider_config.build_engine is True
    assert provider_config.force_rebuild is True
    assert provider_config.use_cuda_graph is True


def test_model_registry_is_d2s_model_mapping_source():
    names = set(ModelRegistry.default().names())

    assert "Distill-Any-Depth-Base" in names
    assert "Depth-Anything-V2-Large" in names
    assert "DA3-LARGE" in names
    assert "depth-anything-indoor-large" in names
    assert "depth-anything-outdoor-large" in names
    assert "DepthPro-Large" in names


def test_zipdepth_base_and_small_keep_separate_variants_and_artifacts():
    registry = ModelRegistry.default()
    base = registry.get("ZipDepth-Base")
    small = registry.get("ZipDepth-Small")

    assert base.model_id == small.model_id == "pablovela5620/zipdepth"
    assert base.family == "zipdepth"
    assert small.family == "zipdepth-npu"
    assert registry.get(base.model_id).name == "ZipDepth-Base"

    base_paths = artifact_paths_for_model("ZipDepth-Base", cache_dir="cache")
    small_paths = artifact_paths_for_model("ZipDepth-Small", cache_dir="cache")
    assert base_paths.model_dir == small_paths.model_dir
    assert base_paths.onnx_fp16_path != small_paths.onnx_fp16_path
    assert base_paths.onnx_fp16_path.name == "zipdepth_base_model_fp16_384x672.onnx"
    assert small_paths.onnx_fp16_path.name == "zipdepth_small_model_fp16_384x672.onnx"


def test_zipdepth_runtime_config_preserves_selected_variant():
    config = DepthRuntimeConfig(
        model_id="ZipDepth-Small",
        export_height=384,
        export_width=677,
    )

    provider_config = depth_provider_config_from_runtime(config)

    assert provider_config.model_id == "pablovela5620/zipdepth"
    assert provider_config.model_name == "ZipDepth-Small"
    assert provider_config.depth_resolution == 384
    assert provider_config.patch_size == 32
    assert _depth_export_size_from_settings(
        {"Depth Model": "ZipDepth-Small", "Depth Resolution": 384}
    ) == (384, 677)


def test_zipdepth_download_cache_uses_variant_checkpoint(monkeypatch, tmp_path):
    import stereo_runtime.depth_provider as depth_provider

    def fake_download(model_id, filename, cache_dir, endpoint):
        path = resolve_model_dir(model_id, cache_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")

    monkeypatch.setattr(depth_provider, "_reachable_hf_endpoints", lambda model_id: ("hf",))
    monkeypatch.setattr(depth_provider, "_download_hf_file_direct", fake_download)
    registry = ModelRegistry.default()

    for name, family in (("ZipDepth-Base", "zipdepth"), ("ZipDepth-Small", "zipdepth-npu")):
        spec = registry.get(name)
        model_dir = ensure_model_downloaded(spec, cache_dir=tmp_path)

        assert spec.family == family
        assert (model_dir / ZIPDEPTH_CHECKPOINT_FILENAMES[family]).is_file()
        assert _model_cache_is_complete(spec, model_dir)


def test_zipdepth_only_exposes_small_to_coreml_and_migraphx():
    assert "zipdepth-base" in DISABLE_COREML_KEYWORDS
    assert "zipdepth-base" in DISABLE_MIGRAPHX_KEYWORDS
    assert not any(keyword in "zipdepth-small" for keyword in DISABLE_COREML_KEYWORDS)
    assert not any(keyword in "zipdepth-small" for keyword in DISABLE_MIGRAPHX_KEYWORDS)


def test_migraphx_is_hidden_for_zipdepth_base_only():
    assert DISABLE_MIGRAPHX_KEYWORDS == ["zipdepth-base"]


def test_gui_model_catalog_matches_v25_model_list_without_legacy_models():
    settings = (APP_ROOT / "settings.yaml").read_text(encoding="utf-8")
    model_block = settings.replace("\r\n", "\n").split("Model List:\n", 1)[1]
    model_block = model_block.split("\nDepth Model:", 1)[0]
    settings_names = re.findall(r"^  ([^ \n:][^\n:]*):$", model_block, flags=re.MULTILINE)

    assert list(GUI_MODEL_CATALOG) == list(V25_GUI_MODEL_NAMES)
    assert settings_names == list(V25_GUI_MODEL_NAMES)
    assert settings.count("  dpt-") == 0
    assert settings.count("  zoedepth-") == 0
    assert settings.count("  depth-ai:") == 0
    assert len(GUI_MODEL_CATALOG) == 37


def test_utils_model_mapping_delegates_to_runtime_registry():
    utils_source = (APP_ROOT / "utils" / "__init__.py").read_text(encoding="utf-8")
    runtime_exports_source = (APP_ROOT / "utils" / "runtime_exports.py").read_text(encoding="utf-8")
    capabilities_source = (APP_ROOT / "stereo_runtime" / "model_capabilities.py").read_text(encoding="utf-8")

    assert "model_name_mapping" in runtime_exports_source
    assert "from .model_registry import ModelRegistry" in capabilities_source
    assert "spec.name: spec.model_id" in capabilities_source
    assert '"Depth-Anything-V2-Small":' not in utils_source
    assert '"Depth-Anything-V2-Small":' not in runtime_exports_source
