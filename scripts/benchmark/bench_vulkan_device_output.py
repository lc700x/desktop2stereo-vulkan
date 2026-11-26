"""Measure Vulkan image output with and without per-frame input staging.

The benchmark keeps the production image shader unchanged.  The steady-state
case writes RGB/depth once into Vulkan storage buffers, then submits the same
prebound buffers to presenter-owned images.  Output images are never mapped or
read back during timing; one optional diagnostic readback at the end compares
the two image paths byte-for-byte.

Example:
    src/python3/bin/python scripts/benchmark/bench_vulkan_device_output.py \
        --input /tmp/d2s-metal-real-frame-tuning/frame-060s-input.png \
        --frames 20 --warmup 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "desktop2stereo"))

from stereo_runtime.vulkan_backend import (  # noqa: E402
    VulkanStereoComputeBackend,
    VulkanStereoImageComputeBackend,
)
from stereo_runtime.vulkan_stereo_pass import (  # noqa: E402
    VULKAN_HOLE_FILL_NONE,
    VulkanLayeredStereoParams,
)
from viewer.vulkan_context import VulkanContext, VulkanContextConfig  # noqa: E402
from viewer.vulkan_descriptors import VulkanStorageBuffer, VulkanStorageImage  # noqa: E402
from viewer.vulkan_resources import VulkanHostImage  # noqa: E402


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": float(statistics.median(ordered)),
        "p95_ms": float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]),
        "mean_ms": float(statistics.fmean(ordered)),
    }


def _load_input(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as image:
        rgb_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).contiguous()
    # A deterministic full-resolution depth proxy keeps the benchmark
    # independent of a model download while exercising the same buffer sizes.
    depth = (
        rgb[:, 0:1] * 0.299
        + rgb[:, 1:2] * 0.587
        + rgb[:, 2:3] * 0.114
    ).contiguous()
    return rgb, depth


def _stage_bytes(buffer: VulkanStorageBuffer, value: torch.Tensor, channels: int) -> None:
    buffer.write_bytes(VulkanStereoComputeBackend._planar_bytes(value, channels=channels))


def _readback_hash(context, image: VulkanStorageImage, width: int, height: int) -> str:
    vk = context.vk
    readback = VulkanHostImage(
        context,
        width,
        height,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
        label="vulkan-device-output-diagnostic",
        readback=True,
    )
    try:
        timeline = context.copy_image(image, readback.resource)
        context.wait_for_timeline(timeline)
        rgba = np.ascontiguousarray(readback.read_rgba())
        return hashlib.sha256(rgba.tobytes()).hexdigest()
    finally:
        readback.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0:
        raise SystemExit("--frames must be positive and --warmup must be non-negative")

    rgb, depth = _load_input(args.input)
    height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
    context = VulkanContext.create(VulkanContextConfig(frame_context_count=3))
    left = right = None
    host_backend = prebound_backend = None
    rgb_buffer = depth_buffer = None
    try:
        vk = context.vk
        usage = vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        left = VulkanStorageImage(
            context,
            width,
            height,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            usage=usage,
        )
        right = VulkanStorageImage(
            context,
            width,
            height,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            usage=usage,
        )
        context.wait_for_timeline(
            max(left.transition_to_general(role="compute"), right.transition_to_general(role="compute"))
        )
        prebound_backend = VulkanStereoImageComputeBackend(context)
        sizes = {
            "rgb": width * height * 3 * 4,
            "depth": width * height * 4,
        }
        rgb_buffer = VulkanStorageBuffer(context, sizes["rgb"])
        depth_buffer = VulkanStorageBuffer(context, sizes["depth"])
        _stage_bytes(rgb_buffer, rgb, 3)
        _stage_bytes(depth_buffer, depth, 1)
        # Match the no-fill Vulkan configuration used in the Metal comparison.
        # The dataclass default enables the much more expensive balanced fill.
        params = VulkanLayeredStereoParams(
            hole_fill_mode=VULKAN_HOLE_FILL_NONE,
            occlusion_enabled=False,
        )

        # Host-upload case: the shader output is still a Vulkan image and is
        # never read back during timing.
        host_samples: list[float] = []
        for iteration in range(args.warmup + args.frames):
            started = time.perf_counter()
            timeline, _debug = prebound_backend.submit_to_images(
                rgb,
                depth,
                left,
                right,
                params=params,
            )
            context.wait_for_timeline(timeline)
            if iteration >= args.warmup:
                host_samples.append((time.perf_counter() - started) * 1000.0)
        host_hash = (
            _readback_hash(context, left, width, height),
            _readback_hash(context, right, width, height),
        )

        # Prebound case: the same static inputs were staged once above. This
        # isolates per-submit upload overhead, not a live per-frame producer.
        prebound_samples: list[float] = []
        for iteration in range(args.warmup + args.frames):
            started = time.perf_counter()
            timeline, _debug = prebound_backend.submit_prebound_to_images(
                width=width,
                height=height,
                rgb_buffer=rgb_buffer,
                depth_buffer=depth_buffer,
                left_eye=left,
                right_eye=right,
                params=params,
            )
            context.wait_for_timeline(timeline)
            if iteration >= args.warmup:
                prebound_samples.append((time.perf_counter() - started) * 1000.0)
        prebound_hash = (
            _readback_hash(context, left, width, height),
            _readback_hash(context, right, width, height),
        )

        report = {
            "input": str(args.input),
            "depth_source": "derived_luma_proxy",
            "shape": [1, 3, height, width],
            "frames": args.frames,
            "warmup": args.warmup,
            "device": str(getattr(getattr(context, "device_info", None), "name", "unknown")),
            "hole_fill_mode": params.hole_fill_mode,
            "occlusion_enabled": params.occlusion_enabled,
            "host_upload_each_frame": _summary(host_samples),
            "prebound_input_no_upload": _summary(prebound_samples),
            "host_upload_elision_ms": _summary(host_samples)["median_ms"]
            - _summary(prebound_samples)["median_ms"],
            "input_buffers_populated_before_timing": True,
            "input_changed_between_measured_frames": False,
            "output_readback_during_timing": False,
            "diagnostic_output_sha256": {
                "host_upload_path": {
                    "left": host_hash[0],
                    "right": host_hash[1],
                },
                "prebound_path": {
                    "left": prebound_hash[0],
                    "right": prebound_hash[1],
                },
                "match": host_hash == prebound_hash,
            },
        }
        print(json.dumps(report, indent=2))
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        for resource in (rgb_buffer, depth_buffer, left, right):
            if resource is not None:
                resource.close()
        if prebound_backend is not None:
            prebound_backend.close()
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
