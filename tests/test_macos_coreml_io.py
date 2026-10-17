from __future__ import annotations

import ctypes
import queue
import threading
from types import SimpleNamespace
from pathlib import Path

import pytest

from capture.types import CapturedFrame
from stereo_runtime.providers.apple.native.coreml_io import (
    NativeCoreMLBusy,
    NativeCoreMLFrame,
    NativeCoreMLIOBridge,
    native_anaglyph_method_id,
    native_output_format_id,
    _NativeResult,
    _NativeWarpConfig,
)
from utils.queue_utils import put_latest
from stereo_runtime._fused_warp_mps import pack_target


_NATIVE_SOURCE = Path(__file__).resolve().parents[1] / (
    "src/desktop2stereo/stereo_runtime/providers/apple/native/macos_coreml_io.mm"
)


def _native_frame(bridge) -> NativeCoreMLFrame:
    return NativeCoreMLFrame(
        bridge=bridge,
        slot=1,
        frame_id=42,
        source_width=1920,
        source_height=1080,
        depth_width=336,
        depth_height=196,
        finite_depth=True,
        nonfinite_count=0,
        preprocess_ms=0.4,
        model_ms=8.0,
        postprocess_ms=1.0,
        input_shared=True,
        output_backing_used=True,
        output_zero_copy=False,
    )


def test_native_result_abi_has_stable_layout() -> None:
    assert ctypes.sizeof(_NativeResult) == 96
    assert _NativeResult.source_width.offset > _NativeResult.slot.offset
    assert _NativeResult.raw_normalize_lo.offset == 80
    assert _NativeResult.normalization_history_reset.offset == 88
    assert ctypes.sizeof(_NativeWarpConfig) == 84
    assert _NativeWarpConfig.edge_aa_enabled.offset == 80


def test_native_output_modes_match_runtime_frame_shapes() -> None:
    expected = {
        "half_sbs": (1920, 1080),
        "full_sbs": (3840, 1080),
        "half_tab": (1920, 1080),
        "full_tab": (1920, 2160),
        "mono": (1920, 1080),
        "depth_map": (1920, 1080),
        "anaglyph": (1920, 1080),
        "interleaved": (1920, 1080),
        "leia": (1920, 1080),
    }
    assert {name: pack_target(1920, 1080, name) for name in expected} == expected
    assert {name: native_output_format_id(name) for name in expected} == {
        name: index for index, name in enumerate(expected)
    }
    assert native_anaglyph_method_id("red_cyan") == 0
    assert native_anaglyph_method_id("green_magenta") == 1
    assert native_anaglyph_method_id("amber_blue") == 2
    assert native_anaglyph_method_id("gray") == 3


def test_native_scalar_kernels_use_one_dimensional_threadgroups() -> None:
    source = _NATIVE_SOURCE.read_text()
    normalize_start = source.index("static void d2s_encode_normalize")
    normalize_end = source.index("static float d2s_half_to_float", normalize_start)
    normalize = source[normalize_start:normalize_end]
    pack_start = source.index("int32_t d2s_coreml_io_pack")
    pack = source[pack_start:]

    assert "dispatchThreads:grid" in normalize
    assert "dispatchThreads:grid" in pack
    assert "threadsPerThreadgroup:MTLSizeMake(w, 1, 1)" in normalize
    assert "threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)" in pack
    assert "dispatchThreadgroups:groups" not in normalize
    assert "dispatchThreadgroups:groups" not in pack
    assert "threadsPerThreadgroup:MTLSizeMake(w, h, 1)" not in normalize
    assert "threadsPerThreadgroup:MTLSizeMake(w, h, 1)" not in pack


def test_native_output_resolution_depth_gaussian_runs_before_warp_without_color_aa() -> None:
    source = _NATIVE_SOURCE.read_text()
    pack_start = source.index("static int32_t d2s_coreml_io_pack_internal")
    pack_end = source.index("int32_t d2s_coreml_io_pack(", pack_start)
    pack = source[pack_start:pack_end]

    gaussian = pack.index("if (global_depth_gaussian)")
    upsample = pack.index("setComputePipelineState:ctx->depth_upsample_pipeline")
    warp = pack.index("[encoder setComputePipelineState:ctx->warp_pipeline]")
    assert "depth_upsample_pipeline" in pack[gaussian:warp]
    assert "depth_global_gaussian_pipeline" in pack[gaussian:warp]
    gaussian_start = source.index("kernel void d2s_filter_depth_global_gaussian")
    gaussian_end = source.index(")D2S_FILTER", gaussian_start)
    gaussian_kernel = source[gaussian_start:gaussian_end]
    assert "float weight = wx * wy;" in gaussian_kernel
    assert "edge_threshold" not in gaussian_kernel
    assert "MTLSizeMake(source_width, source_height, 1)" in pack[gaussian:warp]
    assert "[gaussian_encoder setBuffer:warp_depth_buffer offset:0 atIndex:0]" in pack[gaussian:warp]
    assert gaussian < upsample < warp
    assert "fxaa" not in pack.lower()
    assert "memcpy(destination, destination_buffer.contents, output_bytes);" in pack


def test_native_model_depth_filter_is_separable_and_skips_duplicate_output_aa() -> None:
    source = _NATIVE_SOURCE.read_text()
    filter_start = source.index("kernel void d2s_filter_depth_guided")
    filter_end = source.index("struct DepthUpsampleParams", filter_start)
    model_filter = source[filter_start:filter_end]

    assert "int sx = int(gid.x) + (p.axis == 0 ? offset : 0);" in model_filter
    assert "int sy = int(gid.y) + (p.axis == 1 ? offset : 0);" in model_filter
    assert "smooth_sum / max(weight_sum" in model_filter
    assert "texture2d<float" not in model_filter

    pack_start = source.index("static int32_t d2s_coreml_io_pack_internal")
    pack_end = source.index("int32_t d2s_coreml_io_pack(", pack_start)
    pack = source[pack_start:pack_end]
    assert "for (int axis = 0; axis < 2; ++axis)" in pack
    assert "global_depth_gaussian = global_depth_gaussian && !model_depth_filtered;" in pack
    assert "params.stereo.antialias_strength = -1.0f;" in pack

    sample_start = source.index("static inline float dibr_depth_at(")
    sample_end = source.index("float sx = clamp", sample_start)
    filtered_sample = source[sample_start:sample_end]
    assert "if (p.antialias_strength < 0.0f)" in filtered_sample
    assert "return depth_pop(depth_at(depth, x, y, p), p);" in filtered_sample
    assert "float front_depth = p.antialias_strength != 0.0f" in source


def test_native_depth_upsample_uses_continuous_curvature_confidence() -> None:
    source = _NATIVE_SOURCE.read_text()
    start = source.index("static inline float d2s_upsample_curvature")
    end = source.index("struct DepthGaussianParams", start)
    upsample = source[start:end]

    assert "return max(curvatureX, curvatureY)" in upsample
    for weight, x, y in (("s00", "x0", "y0"), ("s10", "x1", "y0"),
                         ("s01", "x0", "y1"), ("s11", "x1", "y1")):
        assert f"{weight} * d2s_upsample_curvature(source_depth, {x}, {y}," in upsample
    assert "float edgeWeight = smoothstep(" in upsample
    assert "mix(bilinear, guided, edgeWeight * confidence)" in upsample
    assert "maximum_depth - minimum_depth" not in upsample


def test_native_metal_bridge_preserves_metal_logical_rgb_channel_order() -> None:
    source = _NATIVE_SOURCE.read_text()
    preprocess_start = source.index("kernel void d2s_preprocess")
    preprocess_end = source.index("kernel void d2s_normalize_half", preprocess_start)
    preprocess = source[preprocess_start:preprocess_end]
    pack_start = source.index("kernel void d2s_warp_pack")
    pack_end = source.index(")D2S\";", pack_start)
    pack = source[pack_start:pack_end]

    assert "output[offset] = (pixel.r - p.mean0) / p.std0;" in preprocess
    assert "output[plane + offset] = (pixel.g - p.mean1) / p.std1;" in preprocess
    assert "output[2u * plane + offset] = (pixel.b - p.mean2) / p.std2;" in preprocess
    assert "output[offset + 0u] = uchar(clamp(pixel.r * 255.0f" in pack
    assert "pixel.g * 255.0f" in pack
    assert "output[offset + 2u] = uchar(clamp(pixel.b * 255.0f" in pack


def test_native_warp_uses_depth_dimensions_for_depth_sampling() -> None:
    source = _NATIVE_SOURCE.read_text()
    warp_start = source.index("struct WarpParams")
    warp_end = source.index("kernel void d2s_warp_pack", warp_start)
    warp = source[warp_start:warp_end]
    pack_start = source.index("int32_t d2s_coreml_io_pack")
    pack = source[pack_start:]

    assert "uint depth_width;" in warp
    assert "uint depth_height;" in warp
    assert "uint32_t depth_width;" in source
    assert "uint32_t depth_height;" in source
    assert "(uint32_t)ctx->depth_width" in pack
    assert "(uint32_t)ctx->depth_height" in pack
    assert "depth_sample(depth, p.depth_width, p.depth_height" in source
    assert "depth, p.source_width, p.source_height, p" not in source
    assert "layered_warp_at" in warp
    assert "occlusion_at" in warp


def test_native_metal_pack_has_all_display_mode_and_finite_depth_branches() -> None:
    source = _NATIVE_SOURCE.read_text()
    pack_start = source.index("kernel void d2s_warp_pack")
    pack_end = source.index(")D2S\";", pack_start)
    pack = source[pack_start:pack_end]

    assert "p.output_format != 0u" in source
    assert "D2S_OUTPUT_FULL_SBS" in source
    for mode_id in range(2, 9):
        assert f"p.output_format == {mode_id}u" in pack
    assert "output_eye_pixel_vertical" in pack
    assert "finite_color(pixel)" in pack
    assert "isfinite(value) ? value : 1.0f" in pack


def test_native_eye_signs_match_vulkan_layered_path() -> None:
    source = _NATIVE_SOURCE.read_text()
    pack_start = source.index("kernel void d2s_warp_pack")
    pack_end = source.index(")D2S\";", pack_start)
    pack = source[pack_start:pack_end]

    assert "float3 left_pixel = eye_pixel(color, depth, float(x), float(y), 1.0f" in pack
    assert "p.symmetric != 0 ? -1.0f : -0.9f" in pack
    assert "left ? 1.0f : (p.symmetric != 0 ? -1.0f : -0.9f)" in pack


def test_native_coreml_metal_vulkan_pack_is_not_reported_as_fallback() -> None:
    pipeline = Path(__file__).resolve().parents[1] / (
        "src/desktop2stereo/stereo_runtime/pipeline.py"
    )
    source = pipeline.read_text()
    assert 'startswith(\n                "native_coreml_metal"' in source


def test_native_result_reports_finite_depth_contract() -> None:
    frame = _native_frame(object())
    assert frame.finite_depth is True
    assert frame.nonfinite_count == 0


def test_native_frame_pack_uses_destination_without_host_conversion() -> None:
    class RecordingBridge:
        def __init__(self):
            self.pack_calls = []
            self.release_calls = []

        def pack(self, frame, destination, destination_size, width, height, output_format):
            self.pack_calls.append(
                (frame.slot, destination, destination_size, width, height, output_format)
            )

        def release(self, slot):
            self.release_calls.append(slot)

    bridge = RecordingBridge()
    frame = _native_frame(bridge)
    destination = bytearray(2 * 1 * 4)

    frame.pack(destination, (2, 1), "half_sbs")
    frame.release()
    frame.release()

    assert len(bridge.pack_calls) == 1
    assert bridge.pack_calls[0][2] == len(destination)
    assert bridge.release_calls == [1]


def test_native_frame_release_waits_for_in_flight_pack() -> None:
    pack_started = threading.Event()
    finish_pack = threading.Event()
    release_started = threading.Event()
    release_called = threading.Event()
    call_order = []

    class BlockingBridge:
        def pack(self, frame, destination, destination_size, width, height, output_format):
            pack_started.set()
            assert finish_pack.wait(timeout=2)
            call_order.append("pack")

        def release(self, slot):
            call_order.append("release")
            release_called.set()

    frame = _native_frame(BlockingBridge())
    pack_thread = threading.Thread(
        target=frame.pack,
        args=(bytearray(8), (2, 1), "half_sbs"),
    )
    pack_thread.start()
    assert pack_started.wait(timeout=1)

    def release_frame() -> None:
        release_started.set()
        frame.release()

    release_thread = threading.Thread(target=release_frame)
    release_thread.start()
    assert release_started.wait(timeout=1)
    assert not release_called.wait(timeout=0.05)

    finish_pack.set()
    pack_thread.join(timeout=2)
    release_thread.join(timeout=2)

    assert not pack_thread.is_alive()
    assert not release_thread.is_alive()
    assert call_order == ["pack", "release"]


def test_native_frame_can_request_gpu_rgb24_pack() -> None:
    class RecordingBridge:
        def __init__(self):
            self.rgb = None

        def pack(self, frame, destination, destination_size, width, height, output_format, *, rgb=False):
            self.rgb = rgb

        def release(self, slot):
            pass

    bridge = RecordingBridge()
    frame = _native_frame(bridge)
    frame.pack(bytearray(2 * 1 * 3), (2, 1), "half_sbs", rgb=True)

    assert bridge.rgb is True


def test_native_frame_can_write_directly_to_an_exported_metal_buffer() -> None:
    class RecordingBridge:
        def __init__(self):
            self.pack_call = None

        def pack(
            self, frame, destination, destination_size, width, height,
            output_format, *, metal_buffer=None,
        ):
            self.pack_call = (
                destination, destination_size, width, height, output_format,
                metal_buffer.value,
            )
            return {
                "metal_gpu_ms": 0.7,
                "host_handoff_copy_ms": 0.0,
                "metal_buffer_alias": True,
                "warp_threads": 128,
            }

        def release(self, _slot):
            pass

    bridge = RecordingBridge()
    frame = _native_frame(bridge)
    buffer = ctypes.create_string_buffer(64)

    metrics = frame.pack_metal(
        ctypes.addressof(buffer), 64, (4, 2), "half_sbs"
    )

    assert bridge.pack_call == (
        None, 64, 4, 2, "half_sbs", ctypes.addressof(buffer)
    )
    assert metrics["metal_buffer_alias"] is True
    assert metrics["warp_threads"] == 128
    assert frame.last_pack_metrics == metrics


def test_native_frame_accepts_shared_layered_warp_configuration() -> None:
    bridge = object()
    frame = _native_frame(bridge)

    frame.configure_warp(
        depth_strength=0.25,
        max_disparity_px=48.0,
        layers=2,
        edge_dilation=1,
        hole_fill_mode=2,
        occlusion_enabled=1,
    )

    assert frame.warp_config == {
        "depth_strength": 0.25,
        "max_disparity_px": 48.0,
        "layers": 2,
        "edge_dilation": 1,
        "hole_fill_mode": 2,
        "occlusion_enabled": 1,
    }


def test_runtime_native_warp_configuration_uses_vulkan_controls() -> None:
    from stereo_runtime.runtime import _configure_native_coreml_warp

    calls = []

    class Native:
        def configure_warp(self, **values):
            calls.append(values)

    config = SimpleNamespace(
        depth_strength=0.25,
        max_disparity_px=None,
        convergence=0.0,
        parallax_preset="standard",
        edge_threshold=0.04,
        hole_fill="none",
        hole_fill_mode="none",
        hole_fill_radius=3,
        hole_fill_strength=1.0,
        mask_feather_radius=3,
        symmetric=True,
        layers=2,
        foreground_shift_scale=1.0,
        midground_shift_scale=1.0,
        background_shift_scale=1.0,
        edge_dilation=1,
        screen_edge_mask_suppression=0,
        occlusion=True,
        depth_pop=0.0,
        depth_antialias_strength=0.0,
    )
    debug = _configure_native_coreml_warp(Native(), config, width=1920, height=1080)

    assert debug["native_coreml_stereo_postprocess"] == "vulkan_layered_equivalent"
    assert calls[0]["layers"] == 2
    assert calls[0]["hole_fill_mode"] == 2
    assert calls[0]["occlusion_enabled"] == 1
    assert calls[0]["max_disparity_px"] == 48.0


def test_native_busy_code_is_explicit_not_a_cpu_fallback(monkeypatch) -> None:
    pytest.importorskip("objc")

    class Library:
        def d2s_coreml_io_predict(self, *_args):
            return -2

        def d2s_coreml_io_last_error(self, _handle):
            return b"native CoreML resource ring is full"

    bridge = object.__new__(NativeCoreMLIOBridge)
    bridge._library = Library()
    bridge._handle = ctypes.c_void_p(1)
    bridge.last_error = ""
    monkeypatch.setattr("objc.pyobjc_id", lambda _value: 1)

    with pytest.raises(NativeCoreMLBusy):
        bridge.predict(object(), 7)


def test_latest_queue_drop_releases_stale_native_depth() -> None:
    class Owned:
        def __init__(self):
            self.released = 0

        def release(self):
            self.released += 1

    first = Owned()
    second = Owned()
    items = queue.Queue(maxsize=1)

    put_latest(items, SimpleNamespace(viewer_native=first))
    put_latest(items, SimpleNamespace(viewer_native=second))

    assert first.released == 1
    assert second.released == 0


def test_latest_queue_drop_releases_stale_sck_lease() -> None:
    class Owned:
        def __init__(self):
            self.released = 0

        def release(self):
            self.released += 1

    first = Owned()
    second = Owned()
    items = queue.Queue(maxsize=1)

    put_latest(items, CapturedFrame(None, (8, 4), 1.0, sck_zero_copy=first))
    put_latest(items, CapturedFrame(None, (8, 4), 2.0, sck_zero_copy=second))

    assert first.released == 1
    assert second.released == 0
