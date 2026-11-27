from pathlib import Path

import torch
from gui.config import FAMILY_SIZE_TO_MODEL, parse_model_name
from stereo_runtime.depth_provider import AutoDepthProvider, DepthProfileResult, DepthProviderConfig
from stereo_runtime.model_artifacts import _model_cache_is_complete, artifact_paths_for_model
from stereo_runtime.model_impl.custom_models import _flashdepth_config, _moge_point_map_to_depth
from stereo_runtime.model_registry import ModelRegistry
from stereo_runtime.new_model_providers import create_family_provider


def test_gui_parser_keeps_new_family_variants_distinct():
    assert parse_model_name("DepthPro2-S") == ("DepthPro2", "S")
    assert parse_model_name("DepthPro2-H") == ("DepthPro2", "H")
    assert parse_model_name("MoGe3-G") == ("MoGe3", "G")
    assert parse_model_name("FlashDepth-Full") == ("FlashDepth", "Full")
    assert FAMILY_SIZE_TO_MODEL[("DepthPro2", "S")] == "DepthPro2-S"
    assert FAMILY_SIZE_TO_MODEL[("FlashDepth", "Full")] == "FlashDepth-Full"


def test_native_family_registry_variants_and_licenses():
    registry = ModelRegistry.default()

    assert [name for name in registry.names() if name.startswith("DepthPro2-")] == [
        "DepthPro2-S",
        "DepthPro2-B",
        "DepthPro2-L",
        "DepthPro2-H",
    ]
    assert registry.get("DepthPro2-H").research_only is True
    assert registry.get("DepthPro2-H").license_id == "apple-research-only"
    assert registry.get("DepthPro2-H").output_semantics == "relative_inverse_depth"
    assert registry.get("MoGe3-G").output_semantics == "metric_z_depth"
    assert registry.get("FlashDepth-Full").compat_profile == "spatial-temporal-off"


def test_shared_repo_variants_have_distinct_artifact_names():
    depth_s = artifact_paths_for_model("DepthPro2-S", cache_dir="models")
    depth_h = artifact_paths_for_model("DepthPro2-H", cache_dir="models")
    flash_s = artifact_paths_for_model("FlashDepth-S", cache_dir="models")
    flash_full = artifact_paths_for_model("FlashDepth-Full", cache_dir="models")

    assert depth_s.model_dir == depth_h.model_dir
    assert depth_s.onnx_fp16_path.name == "depthpro2_s_model_fp16_288x512.onnx"
    assert depth_h.onnx_fp16_path.name == "depthpro2_h_model_fp16_288x512.onnx"
    assert flash_s.model_dir == flash_full.model_dir
    assert flash_s.onnx_fp16_path.name == "flashdepth_s_model_fp16_294x518.onnx"
    assert flash_full.onnx_fp16_path.name == "flashdepth_full_model_fp16_294x518.onnx"


def test_flashdepth_variant_profiles_select_expected_backbone():
    small = _flashdepth_config("FlashDepth-S")
    large = _flashdepth_config("FlashDepth-L")
    full = _flashdepth_config("FlashDepth-Full")

    assert small["vit_size"] == "vits"
    assert small["hybrid_configs"].use_hybrid is False
    assert large["vit_size"] == "vitl"
    assert large["hybrid_configs"].use_hybrid is False
    assert full["vit_size"] == "vits"
    assert full["hybrid_configs"].use_hybrid is True
    for config in (small, large, full):
        assert config["use_mamba"] is False


def test_depthpro2_cache_requires_model_and_fov_weights(tmp_path: Path):
    spec = ModelRegistry.default().get("DepthPro2-S")
    root = spec.model_dir(tmp_path)

    root.mkdir(parents=True)
    (root / spec.weight_files[0]).write_bytes(b"model")
    assert _model_cache_is_complete(spec, root) is False

    (root / spec.weight_files[1]).write_bytes(b"fov")
    assert _model_cache_is_complete(spec, root) is True


def test_flashdepth_cache_resolves_nested_snapshot(tmp_path: Path):
    spec = ModelRegistry.default().get("FlashDepth-S")
    snapshot = spec.model_dir(tmp_path) / "snapshots" / "revision"
    nested = snapshot / "flashdepth-s"
    nested.mkdir(parents=True)
    (nested / "iter_14001.pth").write_bytes(b"weights")

    assert _model_cache_is_complete(spec, spec.model_dir(tmp_path)) is True


def test_family_provider_reports_compat_profile_and_zero_copy():
    cases = [
        ("depth-pro-2", "lc700x/DepthPro2", "DepthPro2-S", "onnx-inverse-depth"),
        ("moge-3", "Ruicheng/moge-3-vitl", "MoGe3-L", "spatial-refine-off"),
        ("flashdepth", "Eyeline-Labs/FlashDepth", "FlashDepth-S", "spatial-temporal-off"),
    ]
    for family, model_id, model_name, profile in cases:
        provider = create_family_provider(
            family=family,
            model_id=model_id,
            model_name=model_name,
            device="cpu",
            cache_dir="models",
            compat_profile=True,
        )
        assert provider.info.model_profile == profile
        assert provider.info.zero_copy_output is True
        assert provider.info.host_readback_bytes == 0


def test_moge_gpu_solver_is_finite_and_preserves_metric_depth_direction():
    points = torch.zeros(1, 16, 16, 3)
    points[..., 2] = 2.0 + torch.linspace(-0.25, 0.25, 16).view(1, 16, 1)
    points[..., 0] = 0.25
    points[..., 1] = -0.10

    depth = _moge_point_map_to_depth(points)

    assert depth.shape == (1, 16, 16)
    assert torch.isfinite(depth).all()
    assert float(depth.abs().max()) > 0.0


def test_new_family_auto_chain_does_not_append_cpu_fallback():
    provider = AutoDepthProvider(
        DepthProviderConfig(
            backend="auto",
            model_id="Ruicheng/moge-3-vitl",
            model_name="MoGe3-L",
            device="cuda",
            local_files_only=True,
        )
    )

    names = [name for name, _factory in provider._candidate_factories()]

    assert "CPU" not in names


def test_depth_profile_result_reports_zero_copy_metadata():
    result = DepthProfileResult(
        depth=torch.zeros(1, 1, 2, 2),
        preprocess_ms=1.0,
        model_ms=2.0,
        postprocess_ms=0.5,
        model_profile="spatial-refine-off",
        zero_copy_output=True,
        host_readback_bytes=0,
    )

    report = result.to_report()
    assert report["model_profile"] == "spatial-refine-off"
    assert report["zero_copy_output"] is True
    assert report["host_readback_bytes"] == 0
