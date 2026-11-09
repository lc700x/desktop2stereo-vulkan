from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from desktop2stereo.stereo_runtime.providers.intel.d3d11_sbs_surface import (
    D3D11SbsSurface,
    probe_d3d11_sbs_surface,
)


def test_d3d11_sbs_surface_probe_is_safe_without_optional_dll() -> None:
    result = probe_d3d11_sbs_surface()
    assert result["backend"] == "d3d11_sbs_surface"
    assert isinstance(result["available"], bool)
    assert result["zero_copy"] is False
    assert isinstance(result["external_bgra_texture_import"], bool)
    assert isinstance(result["external_texture_zero_copy_ready"], bool)


def test_rgb_to_bgra_preserves_final_sbs_geometry_and_channels() -> None:
    from streaming.direct_sbs import IntelD3D11DirectSbsOutput

    rgb = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    bgra = IntelD3D11DirectSbsOutput._rgb_to_bgra(rgb)
    assert bgra.shape == (1, 2, 4)
    assert bgra.tolist() == [[[3, 2, 1, 255], [6, 5, 4, 255]]]


def test_intel_vulkan_sbs_frame_requires_explicit_producer_readiness() -> None:
    from stereo_runtime.intel_vulkan_sbs import IntelVulkanSbsFrame

    frame = IntelVulkanSbsFrame(1, 2, 64, 32, 99)
    assert frame.gpu_to_cpu is False
    assert frame.zero_copy is False
    assert frame.gpu_copy_count == 1
    assert frame.producer_ready is False
    assert frame.ready_timeline == 0

    ready = IntelVulkanSbsFrame(
        1, 2, 64, 32, 99, producer_ready=True, ready_timeline=7
    )
    assert ready.producer_ready is True
    assert ready.ready_timeline == 7
    assert ready.zero_copy is False


def test_d3d11_surface_requires_optional_bridge() -> None:
    result = probe_d3d11_sbs_surface()
    if result["available"]:
        return
    try:
        D3D11SbsSurface(width=64, height=64)
    except RuntimeError as exc:
        assert "d3d11" in str(exc).lower()
    else:
        raise AssertionError("D3D11 SBS surface unexpectedly initialized without bridge")


@pytest.mark.parametrize("mode", ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB"])
@pytest.mark.parametrize("fit", ["contain", "cover", "stretch"])
def test_native_eye_blits_keep_source_eyes_separate_at_scaled_output(mode, fit):
    from stereo_runtime.intel_vulkan_sbs import _eye_blit_regions

    target = (160 * (2 if mode == "Full-SBS" else 1), 100 * (2 if mode == "Full-TAB" else 1))
    regions = _eye_blit_regions((80, 50), target, mode, fit, (160, 100), packed_source=True)
    assert len(regions) == 2
    for index, (source, destination) in enumerate(regions):
        assert index * 80 <= source[0] < source[2] <= (index + 1) * 80
        assert 0 <= source[1] < source[3] <= 50
        if mode.endswith("SBS"):
            assert index * (target[0] // 2) <= destination[0] < destination[2] <= (index + 1) * (target[0] // 2)
        else:
            assert index * (target[1] // 2) <= destination[1] < destination[3] <= (index + 1) * (target[1] // 2)


def test_native_composer_allocates_encoder_canvas_not_processing_size(monkeypatch):
    import stereo_runtime.intel_vulkan_sbs as bridge

    allocations, submits = [], []
    context = SimpleNamespace(device_info=SimpleNamespace(adapter_luid=99), wait_for_timeline=lambda *_: None)
    context.compose_sbs_images = lambda *args, **kwargs: submits.append((args, kwargs)) or 7

    class Surface:
        def __init__(self, *, width, height, adapter_luid):
            self.width, self.height, self.adapter_luid = width, height, adapter_luid
            self.device, self.bgra_texture = 2, 3
            allocations.append((width, height))

        def import_bgra_into_vulkan(self, ctx):
            return SimpleNamespace(resource=SimpleNamespace(context=ctx), close=lambda: None)

        def close(self):
            pass

    monkeypatch.setattr(bridge, "D3D11SbsSurface", Surface)
    composer = bridge.IntelVulkanSbsComposer(context, 80, 50, target_size=(160, 100))
    left, right = SimpleNamespace(context=context), SimpleNamespace(context=context)
    result = composer.compose(left, right, display_mode="Half-SBS", ready_timeline=4)
    assert allocations == [(160, 100)]
    assert (result.width, result.height) == (160, 100)
    assert result.vulkan_resource is composer.imported.resource
    assert submits[0][1]["wait_for_timeline"] == 4
    assert len(submits[0][1]["blit_regions"]) == 2
    composer.close()
