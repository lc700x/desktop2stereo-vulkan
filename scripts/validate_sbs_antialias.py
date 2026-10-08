"""Production native Metal SBS AA experiments, with frozen source baselines.

Run with src/python3/bin/python scripts/validate_sbs_antialias.py --freeze-baseline.
Images preserve source resolution and native inferred depth across AA variants.
The reference shader is copied for reproducibility; repository sources and user
settings are never replaced or edited by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import re
import struct
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/desktop2stereo"))

DEFAULT_OUT = Path.home() / "Downloads/dibr-edge-review/output-aa-v2"
INPUTS = {"test_jpg": "test.jpg", "test2": "test2.jpeg", "test3": "test3.png"}
CROPS = {
    "test_jpg": {"laptop_top": (300, 260, 645, 345), "laptop_base": (280, 545, 645, 630),
                 "right_laptop": (620, 200, 925, 540)},
    "test2": {"box": (750, 330, 950, 614)},
    "test3": {"box_fixed": (1300, 710, 1338, 970), "box_padded": (1285, 695, 1350, 980),
              "fingers": (1030, 320, 1400, 680), "box_diagonal": (810, 600, 1130, 810)},
}


def percentile_stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"samples": len(values), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


def pixel_buffer_for(rgb):
    from Quartz import CoreVideo as CV
    h, w = rgb.shape[:2]
    status, pb = CV.CVPixelBufferCreate(None, w, h, CV.kCVPixelFormatType_32BGRA,
        {CV.kCVPixelBufferIOSurfacePropertiesKey: {}, CV.kCVPixelBufferMetalCompatibilityKey: True}, None)
    if status or pb is None:
        raise RuntimeError(f"CVPixelBufferCreate failed: {status}")
    status = CV.CVPixelBufferLockBaseAddress(pb, 0)
    if status:
        raise RuntimeError(f"CVPixelBufferLockBaseAddress failed: {status}")
    try:
        bpr = int(CV.CVPixelBufferGetBytesPerRow(pb))
        rows = np.frombuffer(CV.CVPixelBufferGetBaseAddress(pb).as_buffer(bpr * h), dtype=np.uint8).reshape(h, bpr)
        px = rows[:, :w * 4].reshape(h, w, 4)
        px[..., :3] = rgb[..., [2, 1, 0]]
        px[..., 3] = 255
    finally:
        CV.CVPixelBufferUnlockBaseAddress(pb, 0)
    return pb


def freeze_source(output, coreml_io):
    frozen = output / "baseline_source"
    frozen.mkdir(parents=True, exist_ok=True)
    for source in (
        coreml_io._SOURCE,
        coreml_io._HEADER,
        coreml_io._FXAA_HEADER,
    ):
        target = frozen / source.name
        if not target.exists():
            shutil.copyfile(source, target)
    return frozen / coreml_io._SOURCE.name, frozen / coreml_io._HEADER.name


def save_crops(eyes, name, variant, output):
    for eye, image in eyes.items():
        for feature, crop in CROPS[name].items():
            piece = image.crop(crop)
            piece.save(output / f"{variant}_{eye}_{feature}.png")
            piece.resize((piece.width * 4, piece.height * 4), Image.Resampling.NEAREST).save(
                output / f"{variant}_{eye}_{feature}_4x.png")


def save_pair_comparisons(name, folder):
    for fmt in ("full_sbs", "half_sbs"):
        for eye in ("left", "right"):
            off = Image.open(folder / f"current_off_{fmt}_{eye}.png").convert("RGB")
            on = Image.open(folder / f"current_on_{fmt}_{eye}.png").convert("RGB")
            for feature, crop in CROPS[name].items():
                if fmt == "half_sbs":
                    crop = (crop[0] // 2, crop[1], (crop[2] + 1) // 2, crop[3])
                a, b = off.crop(crop), on.crop(crop)
                a = a.resize((a.width * 4, a.height * 4), Image.Resampling.NEAREST)
                b = b.resize((b.width * 4, b.height * 4), Image.Resampling.NEAREST)
                canvas = Image.new("RGB", (a.width + b.width, a.height + 30), "white")
                canvas.paste(a, (0, 30))
                canvas.paste(b, (a.width, 30))
                draw = ImageDraw.Draw(canvas)
                draw.text((6, 7), "Before: output AA off", fill="black")
                draw.text((a.width + 6, 7), "After: shared FXAA", fill="black")
                canvas.save(folder / f"compare_{fmt}_{eye}_{feature}_4x.png")


def benchmark_pack(frame, width, height, args, folder):
    """Paired bounded-time pack samples: inference runs only before this call."""
    output = bytearray(width * height * 4)
    results = []
    for round_index in range(args.steady_rounds):
        for aa in ((0, 1) if round_index % 2 == 0 else (1, 0)):
            frame.warp_config["edge_aa_enabled"] = aa
            for _ in range(16):
                frame.pack(output, (width, height), "half_sbs")
            samples = []
            started = time.perf_counter()
            while time.perf_counter() - started < args.steady_seconds:
                tick = time.perf_counter()
                frame.pack(output, (width, height), "half_sbs")
                samples.append((time.perf_counter() - tick) * 1000)
            elapsed = time.perf_counter() - started
            item = {"round": round_index + 1, "aa_enabled": aa, "elapsed_seconds": elapsed,
                    "frames": len(samples), "uncapped_pack_fps": len(samples) / elapsed,
                    "pack_wall_time": percentile_stats(samples)}
            results.append(item)
            np.savetxt(folder / f"pack_aa{aa}_round{round_index + 1}_samples_ms.csv", samples,
                       delimiter=",", header="pack_ms", comments="")
            print(f"STEADY_PACK_ROUND {json.dumps(item)}", flush=True)
    aggregate = {str(aa): {"fps": sum(row["frames"] for row in results if row["aa_enabled"] == aa) /
                                sum(row["elapsed_seconds"] for row in results if row["aa_enabled"] == aa),
                          "mean_pack_ms": float(np.mean([row["pack_wall_time"]["mean_ms"] for row in results
                                                          if row["aa_enabled"] == aa]))}
                 for aa in (0, 1)}
    summary = {"measurement": "same retained native depth/frame; production Half-SBS pack; model excluded",
               "seconds_per_variant_per_round": args.steady_seconds, "rounds": results, "aggregate": aggregate,
               "pack_delta_ms": aggregate["1"]["mean_pack_ms"] - aggregate["0"]["mean_pack_ms"],
               "uncapped_pack_fps_reduction_percent": 100 * (aggregate["0"]["fps"] - aggregate["1"]["fps"]) / aggregate["0"]["fps"]}
    (folder / "pack_benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def render_real_images(args):
    from Quartz import CoreVideo as CV
    from stereo_runtime.adapter import runtime_config_from_d2s_settings, stereo_config_from_runtime
    from stereo_runtime.depth_provider import _model_input_size
    from stereo_runtime.providers.apple.native import coreml_io
    from stereo_runtime.runtime import _configure_native_coreml_warp
    from utils.bootstrap import bootstrap_settings
    import torch

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    original_source, original_header = coreml_io._SOURCE, coreml_io._HEADER
    if args.freeze_baseline:
        coreml_io._SOURCE, coreml_io._HEADER = freeze_source(output, coreml_io)
        print(f"BASELINE_SOURCE_FROZEN {coreml_io._SOURCE}", flush=True)
    source_hash = hashlib.sha256(coreml_io._SOURCE.read_bytes()).hexdigest()
    settings = dict(bootstrap_settings(str(ROOT / "src/desktop2stereo/settings.yaml"), os_name="Darwin"))
    model_name = {
        "small": "Distill-Any-Depth-Small",
        "base": "Distill-Any-Depth-Base",
    }[args.model]
    package_dir = {
        "small": "xingyang1-Distill-Any-Depth-Small-hf",
        "base": "lc700x-Distill-Any-Depth-Base-hf",
    }[args.model]
    settings["Depth Model"] = model_name
    settings["Depth Resolution"] = args.resolution
    config = runtime_config_from_d2s_settings(settings, cache_dir=str(ROOT / "src/desktop2stereo/models"),
                                             device="mps", depth_only=False)
    stereo = stereo_config_from_runtime(config)
    model_label = f"{model_name} {args.resolution}"
    manifest = {"renderer": "production NativeCoreMLIOBridge Metal pack", "source_sha256": source_hash,
                "source": str(coreml_io._SOURCE), "model": model_label, "images": {}}
    previous_manifest = output / ("baseline_manifest.json" if args.freeze_baseline else "current_manifest.json")
    if previous_manifest.exists():
        manifest["images"] = json.loads(previous_manifest.read_text()).get("images", {})
    shared_header = coreml_io._SOURCE.parent / "sbs_fxaa_msl.h"
    if shared_header.exists():
        manifest["shared_aa_source_sha256"] = hashlib.sha256(shared_header.read_bytes()).hexdigest()
    tags = [("baseline_default", 0), ("baseline_legacy_supersampling", 1)] if args.freeze_baseline else [
        ("current_off", 0), ("current_on", 1)]
    try:
        for name in args.images:
            image = Image.open(Path.home() / "Downloads" / INPUTS[name]).convert("RGB")
            w, h = image.size
            w -= w % 2
            image = image.crop((0, 0, w, h))
            folder = output / name
            folder.mkdir(parents=True, exist_ok=True)
            image.save(folder / "input.png")
            rgb = np.asarray(image, dtype=np.uint8)
            pb = pixel_buffer_for(rgb)
            ih, iw = _model_input_size(h, w, args.resolution, 14)
            model = ROOT / "src/desktop2stereo/models/coreml" / package_dir / f"model_fp32_{ih}x{iw}.mlpackage"
            frame = bridge = None
            try:
                bridge = coreml_io.NativeCoreMLIOBridge(model, iw, ih)
                frame = bridge.predict(pb, 9000)
                _configure_native_coreml_warp(frame, stereo, width=w, height=h)
                item = {"input": str(Path.home() / "Downloads" / INPUTS[name]), "source_size": [w, h],
                        "coreml_input_size": [iw, ih], "warp": dict(frame.warp_config),
                        "normalize_lo": frame.normalize_lo, "normalize_hi": frame.normalize_hi,
                        "model_ms": frame.model_ms, "variants": {}}
                depth = bytearray(w * h * 4)
                frame.pack(depth, (w, h), "depth_map")
                Image.frombytes("RGBA", (w, h), bytes(depth)).convert("RGB").save(folder / "depth.png")
                rendered_outputs = {}
                for tag, aa in tags:
                    frame.warp_config["edge_aa_enabled"] = aa
                    variant = {"aa_enabled": aa, "timings": {}}
                    rendered_outputs[tag] = {}
                    for fmt, size in [("full_sbs", (2 * w, h)), ("half_sbs", (w, h))]:
                        buffer = bytearray(size[0] * size[1] * 4)
                        for _ in range(3):
                            frame.pack(buffer, size, fmt)
                        timings = []
                        for _ in range(args.samples):
                            start = time.perf_counter()
                            frame.pack(buffer, size, fmt)
                            timings.append((time.perf_counter() - start) * 1000)
                        sbs = Image.frombytes("RGBA", size, bytes(buffer)).convert("RGB")
                        rendered_outputs[tag][fmt] = np.asarray(sbs).copy()
                        sbs.save(folder / f"{tag}_{fmt}.png")
                        eye_width = size[0] // 2
                        eyes = {eye: sbs.crop((idx * eye_width, 0, (idx + 1) * eye_width, h))
                                for idx, eye in enumerate(("left", "right"))}
                        for eye, eye_image in eyes.items():
                            eye_image.save(folder / f"{tag}_{fmt}_{eye}.png")
                        if fmt == "full_sbs":
                            save_crops(eyes, name, tag, folder)
                        variant["timings"][fmt] = percentile_stats(timings)
                    item["variants"][tag] = variant
                    print(f"{name} {tag}: {json.dumps(variant['timings'])}", flush=True)
                if not args.freeze_baseline:
                    from stereo_runtime.display_antialias import antialias_sbs, antialias_sbs_half

                    raw_full = torch.from_numpy(
                        rendered_outputs["current_off"]["full_sbs"].copy()
                    ).permute(2, 0, 1).contiguous()
                    expected_full = antialias_sbs(raw_full, "full_sbs")
                    expected_half = antialias_sbs_half(raw_full)
                    expected_full = expected_full.permute(1, 2, 0).cpu().numpy()
                    expected_half = expected_half.permute(1, 2, 0).cpu().numpy()
                    actual_full = rendered_outputs["current_on"]["full_sbs"]
                    actual_half = rendered_outputs["current_on"]["half_sbs"]
                    full_delta = int(np.abs(actual_full.astype(np.int16) - expected_full.astype(np.int16)).max())
                    half_delta = int(np.abs(actual_half.astype(np.int16) - expected_half.astype(np.int16)).max())
                    item["shared_fxaa_parity"] = {
                        "full_sbs_max_channel_delta": full_delta,
                        "half_sbs_max_channel_delta": half_delta,
                        "max_channel_delta": max(full_delta, half_delta),
                        "passes_one_8bit_level": max(full_delta, half_delta) <= 1,
                    }
                    print(f"{name} shared FXAA parity: {json.dumps(item['shared_fxaa_parity'])}", flush=True)
                manifest["images"][name] = item
                if not args.freeze_baseline:
                    save_pair_comparisons(name, folder)
                if args.steady_seconds:
                    item["steady_pack_benchmark"] = benchmark_pack(frame, w, h, args, folder)
                (folder / ("baseline_manifest.json" if args.freeze_baseline else "current_manifest.json")).write_text(
                    json.dumps(item, indent=2) + "\n")
            finally:
                if frame is not None:
                    frame.release()
                if bridge is not None:
                    bridge.close()
                # The PyObjC wrapper owns the Create result. Explicitly calling
                # CVPixelBufferRelease here would release that ownership twice
                # when Python drops the wrapper.
                pb = None
        manifest_path = output / ("baseline_manifest.json" if args.freeze_baseline else "current_manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"REVIEW_OUTPUT_READY {manifest_path}", flush=True)
    finally:
        coreml_io._SOURCE, coreml_io._HEADER = original_source, original_header


class SyntheticMetalRenderer:
    """Run production Metal warp and the shared FXAA postpack kernel."""

    def __init__(self, source_path):
        import Metal
        self.metal = Metal
        self.device = Metal.MTLCreateSystemDefaultDevice()
        source = Path(source_path).read_text()
        msl = re.search(r'R"D2S\((.*?)\)D2S";', source, re.DOTALL)
        if msl is None:
            raise RuntimeError("Embedded production MSL was not found")
        shader_source = msl.group(1)
        self.library, error = self.device.newLibraryWithSource_options_error_(shader_source, None, None)
        if self.library is None:
            raise RuntimeError(f"Production MSL compilation failed: {error}")
        self.warp = self.pipeline("d2s_warp_pack")
        fxaa_path = Path(source_path).parent / "sbs_fxaa_msl.h"
        fxaa_source = re.search(r'R"MSL\((.*?)\)MSL";', fxaa_path.read_text(), re.DOTALL)
        if fxaa_source is None:
            raise RuntimeError("Production FXAA MSL was not found")
        self.fxaa_library, error = self.device.newLibraryWithSource_options_error_(
            fxaa_source.group(1), None, None
        )
        if self.fxaa_library is None:
            raise RuntimeError(f"Production FXAA MSL compilation failed: {error}")
        self.fxaa = self.fxaa_pipeline("d2s_sbs_fxaa")
        self.fxaa_half = self.fxaa_pipeline("d2s_sbs_fxaa_half")
        self.queue = self.device.newCommandQueue()

    def pipeline(self, name):
        fn = self.library.newFunctionWithName_(name)
        if fn is None:
            raise RuntimeError(f"Production function missing: {name}")
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(f"Pipeline creation failed: {error}")
        return pipeline

    def fxaa_pipeline(self, name):
        fn = self.fxaa_library.newFunctionWithName_(name)
        if fn is None:
            raise RuntimeError(f"Production FXAA function missing: {name}")
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(f"FXAA pipeline creation failed for {name}: {error}")
        return pipeline

    def render(self, rgba, depth, *, half=False, aa=False, depth_strength=0.25, max_disparity=32.0):
        m = self.metal
        h, w = rgba.shape[:2]
        ow = w if half else 2 * w
        full_before_half = bool(aa and half)
        raw_width = 2 * w if (aa or not half) else ow
        descriptor = m.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            m.MTLPixelFormatRGBA8Unorm, w, h, False)
        descriptor.setUsage_(m.MTLTextureUsageShaderRead)
        descriptor.setStorageMode_(m.MTLStorageModeShared)
        texture = self.device.newTextureWithDescriptor_(descriptor)
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            m.MTLRegionMake2D(0, 0, w, h), 0, np.ascontiguousarray(rgba).tobytes(), w * 4)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        db = self.device.newBufferWithBytes_length_options_(depth.tobytes(), depth.nbytes,
                                                           m.MTLResourceStorageModeShared)
        raw = self.device.newBufferWithLength_options_(raw_width * h * 4, m.MTLResourceStorageModeShared)
        output = self.device.newBufferWithLength_options_(ow * h * 4, m.MTLResourceStorageModeShared)
        # Exact 116-byte WarpParams layout including the final edge AA field.
        warp_format = 9 if full_before_half else (0 if half else 1)
        values = (w, h, w, h, raw_width, h, warp_format, 4,
                  depth_strength, max_disparity, 0.0, 0.04, 0.0,
                  0, 1, 1, 2, 0.08, 1.15, 1.05, 1.05, 1, 0, 2, 1,
                  0.0, 0.0, 0, 0)
        packed = struct.pack("<8I5f4i4f4i2f2i", *values)
        params = self.device.newBufferWithBytes_length_options_(packed, len(packed),
                                                               m.MTLResourceStorageModeShared)
        cmd = self.queue.commandBuffer()
        encoder = cmd.computeCommandEncoder()
        encoder.setComputePipelineState_(self.warp)
        encoder.setTexture_atIndex_(texture, 0)
        encoder.setBuffer_offset_atIndex_(db, 0, 0)
        encoder.setBuffer_offset_atIndex_(raw, 0, 1)
        encoder.setBuffer_offset_atIndex_(params, 0, 2)
        encoder.dispatchThreads_threadsPerThreadgroup_(m.MTLSizeMake(raw_width * h, 1, 1), m.MTLSizeMake(64, 1, 1))
        encoder.endEncoding()
        destination = raw
        if aa:
            # Same single dispatch and eye-local rules as Metal, Vulkan,
            # Triton, and the tensor fallback.
            aa_values = struct.pack("<6I", raw_width, h, 4, 0, w, 1)
            aa_params = self.device.newBufferWithBytes_length_options_(
                aa_values, len(aa_values), m.MTLResourceStorageModeShared
            )
            encoder = cmd.computeCommandEncoder()
            encoder.setComputePipelineState_(self.fxaa_half if half else self.fxaa)
            encoder.setBuffer_offset_atIndex_(raw, 0, 0)
            encoder.setBuffer_offset_atIndex_(raw, 0, 1)
            encoder.setBuffer_offset_atIndex_(output, 0, 2)
            encoder.setBuffer_offset_atIndex_(output, 0, 3)
            encoder.setBuffer_offset_atIndex_(aa_params, 0, 4)
            encoder.dispatchThreads_threadsPerThreadgroup_(
                m.MTLSizeMake(ow, h, 1), m.MTLSizeMake(64, 1, 1)
            )
            encoder.endEncoding()
            destination = output
        cmd.commit()
        cmd.waitUntilCompleted()
        if cmd.status() != m.MTLCommandBufferStatusCompleted:
            raise RuntimeError(f"Metal execution failed: {cmd.error()}")
        return np.frombuffer(destination.contents().as_buffer(ow * h * 4), dtype=np.uint8).reshape(h, ow, 4).copy()


def encode_srgb(linear):
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055)


def decode_srgb(encoded):
    return np.where(encoded <= 0.04045, encoded / 12.92,
                    ((encoded + 0.055) / 1.055) ** 2.4)


def analytic_reference(w, h, slope, phase, *, half=False, depth_value=0.0, linear_blend=True):
    """8×8 coverage of a continuous shifted line, never an upscaled raster."""
    eye_width = w // 2 if half else w
    yy, xx = np.mgrid[:h, :eye_width].astype(np.float64)
    x_scale = 2.0 if half else 1.0
    xx = (xx + 0.5) * x_scale - 0.5
    layer_weight = 1 / (1 + np.exp((1 - 2 * depth_value) / 0.08))
    effective = 0.875 + 0.125 * layer_weight
    depth_scale = 1.05 + 0.10 * max(2 * depth_value - 1, 0)
    shift = -depth_value * depth_scale * 0.25 * 32 * 0.5 * effective
    offsets = (np.arange(8) + 0.5) / 8 - 0.5
    eyes, masks = [], []
    low, high = 30 / 255, 220 / 255
    for sign in (1, -1):
        coverage = np.zeros((h, eye_width), dtype=np.float64)
        for dy in offsets:
            for dx in offsets:
                distance = xx + dx * x_scale + shift * sign - (w / 2 + slope * (yy + dy - h / 2) + phase)
                coverage += distance >= 0
        coverage /= 64
        if linear_blend:
            intensity = encode_srgb(decode_srgb(low) * (1 - coverage) + decode_srgb(high) * coverage)
        else:
            intensity = low * (1 - coverage) + high * coverage
        eyes.append(np.repeat(intensity[..., None], 3, axis=2) * 255)
        distance = xx + shift * sign - (w / 2 + slope * (yy - h / 2) + phase)
        masks.append((np.abs(distance) < 2.0 * np.sqrt(1 + slope * slope)) & (yy > 8) & (yy < h - 9))
    return np.concatenate(eyes, axis=1), np.concatenate(masks, axis=1)


def run_synthetic(args):
    source_path = args.output_dir / "baseline_source/macos_coreml_io.mm" if args.freeze_baseline else (
        ROOT / "src/desktop2stereo/stereo_runtime/providers/apple/native/macos_coreml_io.mm")
    renderer = SyntheticMetalRenderer(source_path)
    folder = args.output_dir / ("synthetic_baseline" if args.freeze_baseline else "synthetic_current")
    folder.mkdir(parents=True, exist_ok=True)
    w, h = 256, 128
    yy, xx = np.mgrid[:h, :w]
    cases = []
    for depth_value in (0.0, 0.5):
        for slope in (0.25, 0.5, 1.0, 2.0):
            for phase in (0.0, 0.25, 0.5, 0.75):
                mask = xx >= w / 2 + slope * (yy - h / 2) + phase
                source = np.repeat(np.where(mask, 220, 30)[..., None], 4, axis=2).astype(np.uint8)
                source[..., 3] = 255
                depth = np.full((h, w), depth_value, dtype=np.float32)
                for half in (False, True):
                    off = renderer.render(source, depth, half=half, aa=False)
                    on = renderer.render(source, depth, half=half, aa=True)
                    reference, boundary = analytic_reference(w, h, slope, phase, half=half, depth_value=depth_value)
                    encoded_reference, _ = analytic_reference(w, h, slope, phase, half=half, depth_value=depth_value,
                                                               linear_blend=False)
                    stats = {}
                    for label, ref in (("linear_coverage", reference), ("encoded_coverage", encoded_reference)):
                        stats[label] = {tag: float(np.sqrt(np.mean((img[..., :3][boundary].astype(float) - ref[boundary]) ** 2)))
                                        for tag, img in (("off_rmse", off), ("on_rmse", on))}
                    flat = ~boundary
                    # The coverage mask intentionally omits border scanlines;
                    # those omitted boundary pixels are not flat-region tests.
                    flat[:9] = False
                    flat[-9:] = False
                    max_flat_delta = int(np.abs(off[..., :3].astype(np.int16) - on[..., :3].astype(np.int16))[flat].max())
                    cases.append({"depth": depth_value, "slope": slope, "phase": phase, "format": "half" if half else "full",
                                  "flat_max_rgb_delta": max_flat_delta, **stats})
                    if phase == 0.25 and depth_value == 0.0:
                        tag = f"diagonal_slope{str(slope).replace('.', '_')}_{'half' if half else 'full'}"
                        for label, image in (("off", off), ("on", on), ("reference_8x8", reference)):
                            Image.fromarray(np.rint(image[..., :3]).clip(0, 255).astype(np.uint8)).save(folder / f"{tag}_{label}.png")
    # Narrow objects must retain a visible peak on every interior scanline.
    stripe = np.abs(xx - w / 2 - 0.25 * (yy - h / 2)) < 1.0
    thin = np.repeat(np.where(stripe, 220, 30)[..., None], 4, axis=2).astype(np.uint8)
    thin[..., 3] = 255
    thin_stats = {}
    for half in (False, True):
        image = renderer.render(thin, np.zeros((h, w), np.float32), half=half, aa=True)
        eye = image[:, image.shape[1] // 2:, :3]
        thin_stats["half" if half else "full"] = {"min_scanline_peak": int(eye[8:-8].mean(axis=2).max(axis=1).min()),
                                                  "rows_visible": int((eye[8:-8].mean(axis=2).max(axis=1) > 70).sum())}
        Image.fromarray(image).save(folder / f"thin_2px_{'half' if half else 'full'}.png")
    seam_source = np.zeros((h, w, 4), np.uint8)
    seam_source[..., 0] = np.linspace(0, 255, w).astype(np.uint8)[None, :]
    seam_source[..., 2] = 255 - seam_source[..., 0]
    seam_source[..., 3] = 255
    seam_off = renderer.render(seam_source, np.zeros((h, w), np.float32), aa=False)
    seam_on = renderer.render(seam_source, np.zeros((h, w), np.float32), aa=True)
    seam_delta = int(np.abs(seam_off[:, w - 2:w + 2, :3].astype(np.int16) - seam_on[:, w - 2:w + 2, :3].astype(np.int16)).max())
    # A smooth periodic plane measures displacement without conflating edge
    # smoothing with a depth-strength change.
    ramp = np.rint(127.5 + 100 * np.sin(2 * np.pi * xx / 32)).astype(np.uint8)
    plane_source = np.repeat(ramp[..., None], 4, axis=2)
    plane_source[..., 3] = 255
    plane_depth = np.full((h, w), 0.75, np.float32)
    plane_off = renderer.render(plane_source, plane_depth, aa=False)
    plane_on = renderer.render(plane_source, plane_depth, aa=True)
    basis = np.exp(-1j * 2 * np.pi * np.arange(32, 224) / 32)
    reference_wave = plane_source[32:96, 32:224, 0].mean(axis=0)
    reference_phase = np.sum((reference_wave - reference_wave.mean()) * basis)
    expected_shift = -0.75 * 1.1 * 0.25 * 32 * 0.5 * (0.875 + 0.125 / (1 + np.exp((1 - 1.5) / 0.08)))
    plane_stats = {}
    for eye, offset, expected in (("left", 0, expected_shift), ("right", w, -expected_shift)):
        shifts = {}
        for label, image in (("off", plane_off), ("on", plane_on)):
            wave = image[32:96, offset + 32:offset + 224, 0].mean(axis=0)
            phase = np.sum((wave - wave.mean()) * basis)
            shifts[label] = float(np.angle(phase / reference_phase) / (2 * np.pi / 32))
        plane_stats[eye] = {"expected_shift_source_px": float(expected), "observed_shift_source_px": shifts,
                            "aa_disparity_delta_source_px": abs(shifts["on"] - shifts["off"]),
                            "aa_shift_error_source_px": abs(shifts["on"] - expected)}
    zero_output = renderer.render(plane_source, np.zeros((h, w), np.float32), aa=True)
    zero_eye_delta = int(np.abs(zero_output[:, :w, :3].astype(np.int16) - zero_output[:, w:, :3].astype(np.int16)).max())
    aggregates = {}
    for fmt in ("full", "half"):
        relevant = [case for case in cases if case["format"] == fmt]
        aggregates[fmt] = {}
        for label in ("linear_coverage", "encoded_coverage"):
            off = float(np.mean([case[label]["off_rmse"] for case in relevant]))
            on = float(np.mean([case[label]["on_rmse"] for case in relevant]))
            aggregates[fmt][label] = {"off_mean_rmse": off, "on_mean_rmse": on,
                                      "reduction_percent": 100 * (off - on) / off}
    result = {"coverage_reference": "analytic continuous line; 8x8 target-pixel footprint integration",
              "shader": str(source_path), "shared_fxaa_pipeline_active": True,
              "diagonal_aggregate": aggregates, "diagonal_cases": cases, "thin_2px": thin_stats,
              "sbs_seam_max_delta": seam_delta, "flat_plane": plane_stats,
              "zero_depth_left_right_max_delta": zero_eye_delta}
    result["acceptance"] = {
        "full_diagonal_rmse_reduction_at_least_20_percent": aggregates["full"]["linear_coverage"]["reduction_percent"] >= 20,
        "half_diagonal_rmse_reduction_at_least_20_percent": aggregates["half"]["linear_coverage"]["reduction_percent"] >= 20,
        "full_flat_rgb_delta_at_most_one": max(case["flat_max_rgb_delta"] for case in cases if case["format"] == "full") <= 1,
        "thin_2px_visible_each_interior_scanline": all(stats["rows_visible"] == h - 16 for stats in thin_stats.values()),
        "sbs_seam_does_not_mix": seam_delta <= 1,
        "flat_plane_disparity_delta_at_most_point_one_source_pixel": all(
            stats["aa_disparity_delta_source_px"] <= 0.1 for stats in plane_stats.values()),
        "flat_plane_expected_shift_error_at_most_point_one_source_pixel": all(
            stats["aa_shift_error_source_px"] <= 0.1 for stats in plane_stats.values()),
        "zero_depth_has_no_stereo_disparity": zero_eye_delta == 0,
    }
    (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"diagonal_aggregate": aggregates, "thin_2px": thin_stats, "sbs_seam_max_delta": seam_delta}, indent=2), flush=True)
    if args.require_quality and not all(result["acceptance"].values()):
        raise SystemExit("Synthetic quality acceptance did not pass; see metrics.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--freeze-baseline", action="store_true")
    parser.add_argument("--images", nargs="+", choices=list(INPUTS), default=list(INPUTS))
    parser.add_argument("--model", choices=("small", "base"), default="base")
    parser.add_argument("--resolution", type=int, choices=(336, 518), default=518)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--steady-seconds", type=float, default=0.0)
    parser.add_argument("--steady-rounds", type=int, default=3)
    parser.add_argument("--require-quality", action="store_true")
    args = parser.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("Native production tests require macOS Metal/CoreML.")
    if not args.synthetic_only:
        render_real_images(args)
    if args.synthetic or args.synthetic_only:
        run_synthetic(args)


if __name__ == "__main__":
    main()
