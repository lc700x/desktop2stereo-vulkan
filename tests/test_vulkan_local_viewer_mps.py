"""macOS viewer present-path tests: accelerator frames + device-side pack."""

import numpy as np
import pytest
import torch

from viewer.vulkan_local_viewer import (
    frame_to_rgba_bytes,
    pack_frame_to_rgba8,
)

MPS = torch.device("mps") if torch.backends.mps.is_available() else None

pytestmark = pytest.mark.skipif(MPS is None, reason="requires MPS")


def test_frame_to_rgba_bytes_accepts_mps_float_tensor() -> None:
    frame = torch.rand(1, 3, 64, 64, device=MPS)
    pixels, width, height = frame_to_rgba_bytes(frame)
    assert isinstance(pixels, bytes)
    assert (width, height) == (64, 64)
    assert len(pixels) == 64 * 64 * 4


def test_pack_frame_to_rgba8_returns_hwc_uint8() -> None:
    frame = torch.rand(1, 3, 64, 64, device=MPS)
    host, width, height = pack_frame_to_rgba8(frame)
    assert host is not None
    assert (width, height) == (64, 64)
    assert host.shape == (64, 64, 4)
    assert host.dtype == np.uint8
    assert int(host[..., 3].min()) == 255  # opaque alpha


def test_pack_frame_to_rgba8_returns_none_for_cpu_input() -> None:
    frame = torch.rand(1, 3, 16, 16)
    assert pack_frame_to_rgba8(frame) is None
