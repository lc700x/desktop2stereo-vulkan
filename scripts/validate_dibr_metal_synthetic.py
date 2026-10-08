"""Compile and exercise the production Metal DIBR kernel on synthetic edges.

Run with:
    src/python3/bin/python scripts/validate_dibr_metal_synthetic.py [output-dir]
"""
from __future__ import annotations

import importlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if sys.platform != "darwin":
    raise SystemExit("This validation requires the macOS Metal runtime.")

Metal = importlib.import_module("Metal")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    Path(sys.argv[1]).expanduser()
    if len(sys.argv) > 1
    else ROOT / ".tmp/dibr-edge-review/synthetic"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 256, 128
MAX_DISPARITY = 32.0
DEPTH_STRENGTH = 0.25


def production_pipeline():
    source = (
        ROOT
        / "src/desktop2stereo/stereo_runtime/providers/apple/native/macos_coreml_io.mm"
    ).read_text()
    match = re.search(r'R"D2S\((.*?)\)D2S";', source, re.DOTALL)
    if match is None:
        raise RuntimeError("Could not find the embedded production Metal source.")

    device = Metal.MTLCreateSystemDefaultDevice()
    library, error = device.newLibraryWithSource_options_error_(match.group(1), None, None)
    if library is None:
        raise RuntimeError(f"Metal source compilation failed: {error}")
    function = library.newFunctionWithName_("d2s_warp_pack")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    if pipeline is None:
        raise RuntimeError(f"Metal pipeline creation failed: {error}")
    return device, pipeline


def render(device, pipeline, rgb, depth, tag, half_sbs=False):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    texture_descriptor = (
        Metal.MTLTextureDescriptor
        .texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatRGBA8Unorm, WIDTH, HEIGHT, False
        )
    )
    texture_descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
    texture_descriptor.setStorageMode_(Metal.MTLStorageModeShared)
    color_texture = device.newTextureWithDescriptor_(texture_descriptor)
    color_texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
        Metal.MTLRegionMake2D(0, 0, WIDTH, HEIGHT), 0, rgb.tobytes(), WIDTH * 4
    )

    depth_buffer = device.newBufferWithBytes_length_options_(
        depth.tobytes(), depth.nbytes, Metal.MTLResourceStorageModeShared
    )
    output_width = WIDTH if half_sbs else WIDTH * 2
    output_height = HEIGHT
    output_format = 0 if half_sbs else 1
    output_buffer = device.newBufferWithLength_options_(
        output_width * output_height * 4, Metal.MTLResourceStorageModeShared
    )

    # Keep this layout in sync with the MSL WarpParams structure.
    values = (
        WIDTH, HEIGHT, WIDTH, HEIGHT, output_width, output_height, output_format, 4,
        DEPTH_STRENGTH, MAX_DISPARITY, 0.0, 0.04, 0.0,
        0, 1, 1, 2,
        0.08, 1.15, 1.05, 1.05,
        1, 0, 2, 1,
        0.0, 0.0, 0,
    )
    packed_params = struct.pack("<8I5f4i4f4i2fi", *values)
    params_buffer = device.newBufferWithBytes_length_options_(
        packed_params, len(packed_params), Metal.MTLResourceStorageModeShared
    )

    command_buffer = device.newCommandQueue().commandBuffer()
    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setTexture_atIndex_(color_texture, 0)
    encoder.setBuffer_offset_atIndex_(depth_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(output_buffer, 0, 1)
    encoder.setBuffer_offset_atIndex_(params_buffer, 0, 2)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(output_width * output_height, 1, 1),
        Metal.MTLSizeMake(64, 1, 1),
    )
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    if command_buffer.status() != Metal.MTLCommandBufferStatusCompleted:
        raise RuntimeError(f"Metal command failed: {command_buffer.error()}")

    result = np.frombuffer(
        output_buffer.contents().as_buffer(output_width * output_height * 4),
        dtype=np.uint8,
    ).reshape(HEIGHT, output_width, 4).copy()
    Image.fromarray(rgb, "RGBA").save(OUTPUT_DIR / f"{tag}_source.png")
    Image.fromarray(np.rint(depth * 255).astype(np.uint8)).save(
        OUTPUT_DIR / f"{tag}_depth.png"
    )
    output_name = "half_sbs" if half_sbs else "full_sbs"
    Image.fromarray(result, "RGBA").save(OUTPUT_DIR / f"{tag}_{output_name}.png")
    return result


def rgba_pattern(red, green, blue):
    image = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    image[..., 0] = red
    image[..., 1] = green
    image[..., 2] = blue
    image[..., 3] = 255
    return image


def phase_shift(output, reference, start=32, stop=224):
    xs = np.arange(start, stop, dtype=np.float64)
    omega = 2.0 * np.pi / 32.0
    source_signal = reference[32:96, start:stop, 0].astype(np.float64).mean(axis=0)
    output_signal = output[32:96, start:stop, 0].astype(np.float64).mean(axis=0)
    basis = np.exp(-1j * omega * xs)
    source_phase = np.sum((source_signal - source_signal.mean()) * basis)
    output_phase = np.sum((output_signal - output_signal.mean()) * basis)
    return float(np.angle(output_phase / source_phase) / omega)


def main():
    device, pipeline = production_pipeline()
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]

    # Zero-depth must leave both full-resolution eyes unchanged.
    identity_source = rgba_pattern((x * 7 + y * 3) % 256,
                                   (y * 11 + x) % 256,
                                   (x * 5 + y * 13) % 256)
    zero_depth = render(
        device, pipeline, identity_source, np.zeros((HEIGHT, WIDTH), np.float32),
        "zero_depth",
    )
    zero_delta = max(
        int(np.abs(zero_depth[:, :WIDTH, :3].astype(int)
                   - identity_source[:, :, :3].astype(int)).max()),
        int(np.abs(zero_depth[:, WIDTH:, :3].astype(int)
                   - identity_source[:, :, :3].astype(int)).max()),
    )

    # A flat plane validates the expected sub-pixel shift and half-SBS average.
    ramp = np.rint(
        127.5 + 100.0 * np.sin(2.0 * np.pi * np.arange(WIDTH) / 32.0)
    ).astype(np.uint8)
    flat_source = rgba_pattern(ramp[None, :], ramp[None, :], ramp[None, :])
    flat_depth = np.full((HEIGHT, WIDTH), 0.75, dtype=np.float32)
    flat = render(device, pipeline, flat_source, flat_depth, "flat_depth")
    flat_half = render(
        device, pipeline, flat_source, flat_depth, "flat_depth", half_sbs=True
    )
    left_half = np.rint(
        (flat[:, :WIDTH:2, :3].astype(np.float32)
         + flat[:, 1:WIDTH:2, :3].astype(np.float32)) * 0.5
    ).astype(np.uint8)
    right_half = np.rint(
        (flat[:, WIDTH::2, :3].astype(np.float32)
         + flat[:, WIDTH + 1::2, :3].astype(np.float32)) * 0.5
    ).astype(np.uint8)
    expected_half = np.concatenate((left_half, right_half), axis=1)
    half_average_delta = int(
        np.abs(flat_half[:, :, :3].astype(np.int16)
               - expected_half.astype(np.int16)).max()
    )

    value, softness = 0.75, 0.08
    foreground_weight = 1.0 / (1.0 + np.exp((1.0 - 2.0 * value) / softness))
    effective_layer_scale = 0.875 + 0.125 * foreground_weight
    expected_shift = (
        -value * 1.1 * DEPTH_STRENGTH * MAX_DISPARITY * 0.5
        * effective_layer_scale
    )
    left_shift = phase_shift(flat[:, :WIDTH, :], flat_source)
    right_shift = phase_shift(flat[:, WIDTH:, :], flat_source)

    # A foreground depth step must win where its projection overlaps background.
    overlap_source = rgba_pattern(35, 90, 210)
    foreground = (x >= 110) & (x < 126)
    overlap_source[foreground] = (240, 40, 20, 255)
    overlap_depth = np.full((HEIGHT, WIDTH), 0.15, dtype=np.float32)
    overlap_depth[foreground] = 0.95
    overlap = render(device, pipeline, overlap_source, overlap_depth,
                     "foreground_overlap")
    overlap_half = render(device, pipeline, overlap_source, overlap_depth,
                         "foreground_overlap", half_sbs=True)
    foreground_red = int(overlap[HEIGHT // 2, WIDTH + 106, 0])
    foreground_red_half = int(overlap_half[HEIGHT // 2, WIDTH // 2 + 53, 0])

    # A two-pixel foreground stripe must survive full and half output.
    stripe_source = rgba_pattern(18, 24, 30)
    stripe = (x >= 126) & (x < 128)
    stripe_source[stripe] = (235, 235, 235, 255)
    stripe_depth = np.full((HEIGHT, WIDTH), 0.18, dtype=np.float32)
    stripe_depth[stripe] = 0.94
    stripe_full = render(device, pipeline, stripe_source, stripe_depth,
                         "thin_two_pixel_stripe")
    stripe_half = render(device, pipeline, stripe_source, stripe_depth,
                         "thin_two_pixel_stripe", half_sbs=True)
    stripe_full_pixels = int(
        np.count_nonzero(stripe_full[HEIGHT // 2, WIDTH:, :3].mean(axis=1) > 100)
    )
    stripe_half_pixels = int(
        np.count_nonzero(stripe_half[HEIGHT // 2, WIDTH // 2:, :3].mean(axis=1) > 100)
    )

    # Diagonal occlusion must produce finite colors without holes or NaNs.
    diagonal_source = rgba_pattern(20, 30, 40)
    diagonal_mask = x > WIDTH // 2 + (y - HEIGHT // 2) // 3
    diagonal_source[diagonal_mask] = (220, 180, 110, 255)
    diagonal_depth = np.where(diagonal_mask, 0.92, 0.18).astype(np.float32)
    diagonal = render(device, pipeline, diagonal_source, diagonal_depth,
                      "diagonal_occlusion")

    checks = {
        "zero_depth_max_rgb_delta": zero_delta,
        "flat_depth_expected_shift_source_px": expected_shift,
        "flat_depth_observed_left_shift_source_px": left_shift,
        "flat_depth_left_error_source_px": abs(left_shift - expected_shift),
        "flat_depth_observed_right_shift_source_px": right_shift,
        "flat_depth_right_error_source_px": abs(right_shift + expected_shift),
        "half_sbs_pair_average_max_rgb_delta": half_average_delta,
        "foreground_overlap_full_right_eye_red_at_x106": foreground_red,
        "foreground_overlap_half_right_eye_red_at_x53": foreground_red_half,
        "thin_stripe_full_right_eye_pixels_over_100": stripe_full_pixels,
        "thin_stripe_half_right_eye_pixels_over_100": stripe_half_pixels,
        "diagonal_no_nonfinite_values": bool(np.isfinite(diagonal).all()),
    }
    passed = bool(
        zero_delta <= 1
        and abs(left_shift - expected_shift) <= 0.1
        and abs(right_shift + expected_shift) <= 0.1
        and half_average_delta <= 1
        and foreground_red > 160
        and foreground_red_half > 160
        and stripe_full_pixels >= 2
        and stripe_half_pixels >= 1
        and np.isfinite(diagonal).all()
    )
    metrics = {
        "renderer": "production MSL d2s_warp_pack kernel",
        "resolution": [WIDTH, HEIGHT],
        "depth_strength": DEPTH_STRENGTH,
        "max_disparity_px": MAX_DISPARITY,
        "edge_threshold": 0.04,
        "checks": checks,
        "passed": passed,
        "artifacts": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
