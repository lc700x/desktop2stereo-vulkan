from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch
from gui.model_catalog import GUI_MODEL_CATALOG
from stereo_runtime.depth_provider import (
    AutoDepthProvider,
    DepthProviderConfig,
)
from stereo_runtime.model_registry import ModelRegistry
from stereo_runtime.providers.apple.coreml_families import (
    CoreMLFamilyAccelerationError,
    compute_plan_passes,
    create_coreml_family_provider,
    depth_to_d2s_response,
    summarize_compute_plan_usages,
)
from stereo_runtime.providers.apple.native.coreml_io import _NativeNormalization


class MLCPUComputeDevice:
    pass


class MLGPUComputeDevice:
    pass


class MLNeuralEngineComputeDevice:
    pass


def _usage(preferred, supported, name="op"):
    return SimpleNamespace(
        preferred_compute_device=preferred,
        supported_compute_devices=supported,
        operator_name=name,
    )


def test_gpu_compute_plan_requires_every_operation_to_prefer_gpu():
    report = summarize_compute_plan_usages(
        [
            _usage(MLGPUComputeDevice(), [MLCPUComputeDevice(), MLGPUComputeDevice()]),
            _usage(MLGPUComputeDevice(), [MLCPUComputeDevice(), MLGPUComputeDevice()]),
        ]
    )

    assert compute_plan_passes(report, "gpu") is True
    assert compute_plan_passes(report, "ane") is False


def test_compute_plan_rejects_cpu_only_and_mixed_preference():
    cpu_only = summarize_compute_plan_usages(
        [_usage(MLCPUComputeDevice(), [MLCPUComputeDevice()], "cast")]
    )
    mixed = summarize_compute_plan_usages(
        [
            _usage(MLGPUComputeDevice(), [MLGPUComputeDevice()]),
            _usage(MLCPUComputeDevice(), [MLCPUComputeDevice(), MLGPUComputeDevice()]),
        ]
    )

    assert compute_plan_passes(cpu_only, "gpu") is False
    assert cpu_only["cpu_only_operators"] == ["cast"]
    assert compute_plan_passes(mixed, "gpu") is False


def test_official_depth_outputs_share_distill_near_high_direction():
    metric_z = torch.tensor([1.0, 2.0, 4.0, 0.0])
    response = depth_to_d2s_response(metric_z, "metric_z_depth")
    inverse = depth_to_d2s_response(torch.tensor([1.0, 2.0, 4.0]), "relative_inverse_depth")

    assert response.tolist()[:3] == [1.0, 0.5, 0.25]
    assert response[3] == 200.0
    assert inverse.tolist() == [1.0, 2.0, 4.0]
    assert response[0] > response[1] > response[2]


def test_registry_marks_new_family_variants_as_coreml_capable():
    registry = ModelRegistry.default()
    names = [
        "DepthPro2-S",
        "DepthPro2-B",
        "DepthPro2-L",
        "DepthPro2-H",
        "MoGe3-L",
        "MoGe3-G",
        "FlashDepth-S",
        "FlashDepth-L",
        "FlashDepth-Full",
    ]

    assert all("coreml" in registry.get(name).full_fidelity_backends for name in names)


def test_gui_resolution_lists_follow_model_patch_alignment():
    for name in ("DepthPro2-S", "DepthPro2-B", "DepthPro2-L", "DepthPro2-H"):
        assert all(value % 32 == 0 for value in GUI_MODEL_CATALOG[name]["resolutions"])
        assert ModelRegistry.default().get(name).default_resolution == 512
    for name in ("MoGe3-L", "MoGe3-G", "FlashDepth-S", "FlashDepth-L", "FlashDepth-Full"):
        assert all(value % 14 == 0 for value in GUI_MODEL_CATALOG[name]["resolutions"])
        assert ModelRegistry.default().get(name).default_resolution == 518


def test_macos_mps_family_route_selects_strict_coreml_provider(monkeypatch, tmp_path):
    calls = {}

    def fake_create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(info=SimpleNamespace())

    monkeypatch.setattr(
        "stereo_runtime.providers.apple.create_coreml_family_provider",
        fake_create,
    )
    config = DepthProviderConfig(
        backend="auto",
        model_id="Ruicheng/moge-3-vitl",
        model_name="MoGe3-L",
        device="mps",
        cache_dir=tmp_path,
        use_coreml=True,
    )

    AutoDepthProvider(config)._torch_provider("mps")

    assert calls["family"] == "moge-3"
    assert calls["use_coreml"] is True
    assert calls["coreml_compute_units"] == "gpu"


def test_coreml_provider_load_never_loads_torch_model(tmp_path):
    provider = create_coreml_family_provider(
        family="flashdepth",
        model_id="Eyeline-Labs/FlashDepth",
        model_name="FlashDepth-S",
        device="mps",
        cache_dir=tmp_path,
    )

    assert provider.load() is None
    assert provider._model is None


def test_native_normalization_abi_matches_c_header():
    assert ctypes.sizeof(_NativeNormalization) == 24


def test_provider_reports_explicit_coreml_semantics_fields():
    provider = create_coreml_family_provider(
        family="moge-3",
        model_id="Ruicheng/moge-3-vitl",
        model_name="MoGe3-L",
        device=torch.device("mps"),
        cache_dir="models",
        coreml_compute_units="all",
    )

    assert provider.coreml_compute_units == "gpu"
    assert provider.info.coreml_compute_units is None


def test_family_provider_rejects_non_gpu_accelerator(tmp_path):
    with pytest.raises(CoreMLFamilyAccelerationError, match="GPU-only"):
        create_coreml_family_provider(
            family="flashdepth",
            model_id="Eyeline-Labs/FlashDepth",
            model_name="FlashDepth-S",
            device=torch.device("mps"),
            cache_dir=tmp_path,
            coreml_compute_units="ane",
        )


def test_only_depthpro2_s_enables_native_temporal_stabilization(tmp_path):
    depthpro2 = create_coreml_family_provider(
        family="depth-pro-2",
        model_id="lc700x/DepthPro2",
        model_name="DepthPro2-S",
        device=torch.device("mps"),
        cache_dir=tmp_path,
    )
    flashdepth = create_coreml_family_provider(
        family="flashdepth",
        model_id="Eyeline-Labs/FlashDepth",
        model_name="FlashDepth-S",
        device=torch.device("mps"),
        cache_dir=tmp_path,
    )

    assert depthpro2.native_temporal_history_weight() == 0.85
    assert flashdepth.native_temporal_history_weight() == 0.0
