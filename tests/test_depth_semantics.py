import pytest
import torch
from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
from stereo_runtime.depth_provider import (
    DepthProfileResult,
    DepthProviderInfo,
    _convert_depth_output_to_dibr_response,
    _model_output_semantics,
    _postprocess_model_depth,
)
from stereo_runtime.model_registry import ModelRegistry


@pytest.mark.parametrize(
    ("model_name", "semantics"),
    [
        ("Distill-Any-Depth-Base", "relative_inverse_depth"),
        ("Depth-Anything-V2-Large", "relative_inverse_depth"),
        ("InfiniDepth-Base", "raw_disparity"),
        ("ZipDepth-Base", "relative_inverse_depth"),
        ("Video-Depth-Anything-Base", "relative_inverse_depth"),
        ("Metric-Video-Depth-Anything-Base", "metric_z_depth"),
        ("DA3-BASE", "relative_z_depth"),
        ("DA3MONO-LARGE", "relative_z_depth"),
        ("DA3METRIC-LARGE", "metric_z_depth"),
        ("DA3NESTED-GIANT-LARGE", "metric_z_depth"),
        ("Depth-Anything-V2-Metric-Indoor-Base", "metric_z_depth"),
        ("dpt-dinov2-base-kitti", "metric_z_depth"),
        ("DepthPro-Large", "metric_z_depth"),
    ],
)
def test_registry_declares_official_depth_output_semantics(model_name, semantics):
    assert ModelRegistry.default().get(model_name).output_semantics == semantics
    assert _model_output_semantics(model_name) == semantics


@pytest.mark.parametrize(
    "model_spec",
    ModelRegistry.default().list(),
    ids=lambda spec: spec.name,
)
def test_each_registered_model_semantics_sanitizes_depth_and_preserves_dibr_direction(model_spec):
    width = 64
    semantics = model_spec.output_semantics
    if semantics in {"relative_z_depth", "metric_z_depth"}:
        raw_depth = torch.full((1, 1, 8, width), 1.6)
        raw_depth[..., :16] = 1.0
        raw_depth[..., 48:] = 4.0
    else:
        raw_depth = torch.full((1, 1, 8, width), 0.625)
        raw_depth[..., :16] = 1.0
        raw_depth[..., 48:] = 0.25

    # Keep malformed model samples in the input so every registered semantic
    # path exercises the shared finite-value sanitization before DIBR.
    raw_depth[..., 0, 0] = torch.nan
    raw_depth[..., 0, 1] = torch.inf
    raw_depth[..., 0, 2] = -torch.inf

    depth = _postprocess_model_depth(raw_depth, model_spec.name)
    shift = compute_shift_px(
        depth,
        width,
        ShiftParams(depth_strength=1.0, convergence=0.5, max_disparity_px=40.0),
    )

    assert torch.isfinite(depth).all(), model_spec.name
    assert float(depth.min()) >= 0.0, model_spec.name
    assert float(depth.max()) <= 1.0, model_spec.name
    assert torch.isfinite(shift).all(), model_spec.name
    assert shift[0, 0, 4, 8] < 0, model_spec.name
    assert abs(float(shift[0, 0, 4, 32])) < 1e-5, model_spec.name
    assert shift[0, 0, 4, 56] > 0, model_spec.name


def test_unknown_model_does_not_use_identifier_substring_heuristics():
    assert _model_output_semantics("custom/metric-looking-model") == "relative_inverse_depth"


def test_z_depth_is_inverted_before_normalization():
    inverse_depth = torch.tensor([[[[1.0, 0.5, 0.25]]]])
    z_depth = torch.tensor([[[[1.0, 2.0, 4.0]]]])

    expected = _postprocess_model_depth(inverse_depth, "Distill-Any-Depth-Base")
    actual = _postprocess_model_depth(z_depth, "DA3-BASE")

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual[0, 0, 0, 0] > actual[0, 0, 0, 2]


@pytest.mark.parametrize(
    "model_id",
    [
        "lc700x/DepthPro2",
        "Ruicheng/moge-3-vitl",
        "Ruicheng/moge-3-vitg",
        "Eyeline-Labs/FlashDepth",
    ],
)
def test_shared_new_model_ids_keep_declared_semantics(model_id):
    expected = ModelRegistry.default().get(model_id).output_semantics
    assert _model_output_semantics(model_id) == expected


def test_moge_onnx_wrapper_keeps_training_mask_semantics(monkeypatch):
    from stereo_runtime import onnx_export
    from stereo_runtime.model_impl import custom_models

    calls = {}

    def fake_predict(model, pixel_values, *, target_resolution, apply_mask=True):
        calls.update(
            model=model,
            pixel_values=pixel_values,
            target_resolution=target_resolution,
            apply_mask=apply_mask,
        )
        return pixel_values

    monkeypatch.setattr(custom_models, "predict_moge_3_depth", fake_predict)
    wrapper = onnx_export.NativeMoGeOnnxExportWrapper(object(), target_resolution=518)
    pixel_values = torch.zeros(1, 3, 14, 14)

    assert wrapper(pixel_values) is pixel_values
    assert calls["target_resolution"] == 518
    assert calls["apply_mask"] is True


def test_raw_disparity_is_not_inverted():
    disparity = torch.tensor([[[[1.0, 0.5, 0.25]]]])

    converted = _convert_depth_output_to_dibr_response(disparity, "raw_disparity")

    assert torch.equal(converted, disparity)


def test_dibr_near_far_direction_remains_left_right_symmetric_for_z_depth():
    width = 64
    # Convergence is defined in the normalized DIBR response, so choose the
    # middle region at z=1.6: reciprocal(z) is halfway between the endpoint
    # inverse-depth responses for z=1 and z=4.
    z_depth = torch.full((1, 1, 8, width), 1.6)
    z_depth[..., :16] = 1.0
    z_depth[..., 48:] = 4.0
    depth = _postprocess_model_depth(z_depth, "DA3-BASE")

    shift = compute_shift_px(
        depth,
        width,
        ShiftParams(depth_strength=1.0, convergence=0.5, max_disparity_px=40.0),
    )

    assert shift[0, 0, 4, 8] < 0
    assert abs(float(shift[0, 0, 4, 32])) < 1e-5
    assert shift[0, 0, 4, 56] > 0


def test_provider_reports_raw_and_dibr_semantics():
    info = DepthProviderInfo(
        provider="test",
        model_name="DA3-BASE",
        model_id="depth-anything/DA3-BASE",
        depth_resolution=518,
        cache_dir="",
    )
    result = DepthProfileResult(
        depth=torch.zeros(1, 1, 2, 2),
        preprocess_ms=0.0,
        model_ms=0.0,
        postprocess_ms=0.0,
        raw_output_semantics=info.raw_output_semantics,
    )

    assert info.raw_output_semantics == "relative_z_depth"
    assert info.dibr_output_semantics == "relative_inverse_depth"
    assert result.to_report()["raw_output_semantics"] == "relative_z_depth"
    assert result.to_report()["dibr_output_semantics"] == "relative_inverse_depth"
