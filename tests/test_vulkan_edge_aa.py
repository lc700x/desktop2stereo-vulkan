"""Optional real-pixel check: D2S_TEST_VULKAN_GPU=1 pytest tests/test_vulkan_edge_aa.py."""

import os

import numpy as np
import pytest
import torch

from stereo_runtime.display_antialias import fxaa_reference
from stereo_runtime.vulkan_stereo_image_pass import VulkanStereoImagePass
from stereo_runtime.vulkan_stereo_pass import VulkanLayeredStereoParams, VulkanLayeredStereoPass
from viewer.vulkan_context import VulkanContext
from viewer.vulkan_descriptors import VulkanStorageBuffer, VulkanStorageImage
from viewer.vulkan_resources import VulkanHostReadbackBuffer


@pytest.mark.skipif(os.environ.get("D2S_TEST_VULKAN_GPU") != "1", reason="explicit Vulkan GPU test")
@pytest.mark.parametrize("packed_output", (False, True))
def test_vulkan_fxaa_buffer_and_linear_image_match_with_zero_depth(monkeypatch, packed_output):
    monkeypatch.setenv("D2S_SBS_AA", "1")
    context = VulkanContext.create()
    resources = []
    passes = []
    width, height = 96, 64
    yy, xx = np.mgrid[:height, :width]
    red_edge = (xx > 20 + yy * 0.4).astype(np.float32)
    green_edge = (xx > 30 + yy * 0.25).astype(np.float32)
    blue_edge = (yy >= height // 2).astype(np.float32)
    rgb = np.stack((red_edge, green_edge, blue_edge), axis=0)
    params = VulkanLayeredStereoParams(depth_strength=0.0, hole_fill_mode=2)
    try:
        buffer_pass = VulkanLayeredStereoPass(context, width=width, height=height)
        passes.append(buffer_pass)
        buffers = [VulkanStorageBuffer(context, size) for size in buffer_pass.buffer_sizes.values()]
        resources.extend(buffers)
        buffers[0].write_bytes(rgb.tobytes())
        buffers[1].write_bytes(np.zeros((height, width), np.float32).tobytes())
        # Generic callers receive raw completed eyes for the shared AA seam.
        timeline = buffer_pass.submit(*buffers, params=params, frame_id=0, config_version=1)
        context.wait_for_timeline(timeline)
        raw = np.frombuffer(buffers[2].read_bytes(), np.float32).reshape(3, height, width)
        assert np.array_equal(raw, rgb)
        # Four frames cross the three-slot scratch ring, exercising reuse.
        for frame in range(4):
            timeline = buffer_pass.submit(*buffers, params=params, frame_id=frame,
                config_version=1, apply_edge_aa=True)
            context.wait_for_timeline(timeline)
        encoded = np.frombuffer(buffers[2].read_bytes(), np.float32).reshape(3, height, width)
        expected_encoded = fxaa_reference(torch.from_numpy(rgb).unsqueeze(0))[0].numpy()
        assert np.max(np.abs(encoded - expected_encoded)) <= 1 / 255
        assert np.count_nonzero((encoded > 0.001) & (encoded < 0.999)) > 100
        assert np.array_equal(encoded[:, :8, :8], rgb[:, :8, :8])
        assert encoded.min() >= 0.0 and encoded.max() <= 1.0

        image_pass = VulkanStereoImagePass(context, width=width, height=height, packed_output=packed_output)
        passes.append(image_pass)
        image_width = width * (2 if packed_output else 1)
        images = [VulkanStorageImage(context, image_width, height,
            usage=context.vk.VK_IMAGE_USAGE_STORAGE_BIT | context.vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
            for _ in range(1 if packed_output else 2)]
        resources.extend(images)
        for image in images:
            image.transition_to_general()
        eyes = (images[0], images[0]) if packed_output else tuple(images)
        for frame in range(4):
            timeline = image_pass.submit(*buffers[:2], *eyes, params=params,
                frame_id=frame, config_version=1)
            context.wait_for_timeline(timeline)
        readback = VulkanHostReadbackBuffer(context, image_width, height, label="edge-AA-test")
        resources.append(readback)
        timeline = context.copy_image_to_host_buffer(images[0], readback, wait_for_timeline=timeline)
        context.wait_for_timeline(timeline)
        pixels = readback.read_rgba()
        if packed_output:
            assert np.array_equal(pixels[:, :width], pixels[:, width:])
        actual = pixels[:, :width, :3].transpose(2, 0, 1).astype(np.float32) / 255.0
        expected_linear = np.where(encoded <= 0.04045, encoded / 12.92,
            ((encoded + 0.055) / 1.055) ** 2.4)
        assert np.max(np.abs(expected_linear - actual)) <= 0.5 / 255.0 + 1e-5
    finally:
        context.wait_idle()
        for render_pass in passes:
            render_pass.close()
        for resource in resources:
            resource.close()
        context.close()
