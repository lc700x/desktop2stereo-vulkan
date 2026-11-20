import pytest
import torch
from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
from stereo_runtime.depth_provider import (
    ZipDepthProvider,
    _model_input_size_for_model,
    _postprocess_generic_depth,
)
from stereo_runtime.model_impl.zipdepth import create_model


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (1080, 1920, (384, 672)),
        (1440, 1080, (512, 384)),
        (1920, 1080, (672, 384)),
    ],
)
def test_zipdepth_resizes_short_edge_and_aligns_to_32(height, width, expected):
    assert _model_input_size_for_model(
        height,
        width,
        target=384,
        patch=32,
        model_id="pablovela5620/zipdepth",
    ) == expected


def test_zipdepth_input_resize_antialiases_before_inference(monkeypatch):
    provider = ZipDepthProvider(
        model_id="pablovela5620/zipdepth",
        model_name="ZipDepth-Small",
        device="cpu",
        depth_resolution=32,
        patch_size=32,
    )
    provider._model = _FirstChannelDepth()
    original_interpolate = torch.nn.functional.interpolate
    resize_calls = []

    def track_interpolate(input, *args, **kwargs):
        if input.shape[1] == 3:
            resize_calls.append(kwargs.get("antialias"))
        return original_interpolate(input, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "interpolate", track_interpolate)

    provider.predict_profile(torch.rand(1, 3, 64, 96))

    assert resize_calls and resize_calls[0] is True


def test_zipdepth_base_and_npu_architectures_select_different_upsamplers():
    base = create_model(variant="base", upsample_unfold=True)
    assert base.decoder.convex_up.use_unfold is True
    del base

    small = create_model(variant="base", upsample_unfold=False)
    assert small.decoder.convex_up.use_unfold is False


def test_zipdepth_inverse_depth_is_preserved_and_non_finite_values_are_removed():
    inverse_depth = torch.tensor([[[[1.0, 0.5, 0.25, float("nan")]]]])

    depth = _postprocess_generic_depth(inverse_depth, "pablovela5620/zipdepth")

    assert torch.isfinite(depth).all()
    assert depth[0, 0, 0, 0] > depth[0, 0, 0, 2]
    assert depth.min() >= 0.0
    assert depth.max() <= 1.0




@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zipdepth_inverse_depth_maps_directly_to_linear_dibr_shift(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("accelerator is not available")

    width = 64
    inverse_depth = torch.full((1, 1, 24, width), 0.5, device=device)
    inverse_depth[..., :16] = 0.0
    inverse_depth[..., 32:] = 1.0
    depth = _postprocess_generic_depth(inverse_depth, "pablovela5620/zipdepth")
    params = ShiftParams(depth_strength=0.48, convergence=0.5, max_disparity_px=96.0)

    optimized_shift = compute_shift_px(depth, width, params)

    assert depth[0, 0, 12, 8].item() == pytest.approx(0.0)
    assert depth[0, 0, 12, 24].item() == pytest.approx(0.5)
    assert depth[0, 0, 12, 48].item() == pytest.approx(1.0)
    assert optimized_shift[0, 0, 12, 8].item() == pytest.approx(11.52)
    assert optimized_shift[0, 0, 12, 24].item() == pytest.approx(0.0)
    assert optimized_shift[0, 0, 12, 48].item() == pytest.approx(-11.52)

    monkeypatch.setenv("STEREO_RUNTIME_DISABLE_TRITON", "1")
    torch_shift = compute_shift_px(depth, width, params)
    assert torch.allclose(optimized_shift, torch_shift, atol=2e-5, rtol=2e-5)


class _FirstChannelDepth(torch.nn.Module):
    def forward(self, pixel_values):
        assert torch.isfinite(pixel_values).all()
        return pixel_values[:, :1]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zipdepth_provider_keeps_inference_on_selected_device_and_returns_finite_depth(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("accelerator is not available")

    provider = ZipDepthProvider(
        model_id="pablovela5620/zipdepth",
        model_name="ZipDepth-Small",
        device=device,
        depth_resolution=32,
        patch_size=32,
    )
    provider._model = _FirstChannelDepth()
    rgb = torch.rand(1, 3, 64, 96, device=device)
    rgb[0, 0, 0, 0] = float("nan")
    rgb[0, 0, 0, 1] = float("inf")

    result = provider.predict_profile(rgb)

    assert result.depth.device.type == torch.device(device).type
    assert result.depth.shape == (1, 1, 64, 96)
    assert torch.isfinite(result.depth).all()
