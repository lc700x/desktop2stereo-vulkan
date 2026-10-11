from __future__ import annotations

import struct

from stereo_runtime.vulkan_stereo_pass import VulkanLayeredStereoParams


def test_vulkan_legacy_output_aa_flag_stays_disabled() -> None:
    packed = VulkanLayeredStereoParams().pack(
        1920, 1080, edge_aa_enabled=True
    )

    assert struct.unpack_from("<I", packed, len(packed) - 4)[0] == 0
