"""Registry connecting the packer thread to the Vulkan viewer's staging.

The last non-zero-copy hop in the macOS Vulkan present path was a CPU
memcpy of every packed frame into the persistently mapped staging buffer
before ``vkCmdCopyBufferToImage``. When the mapped pages are 16K-aligned
(they are, on MoltenVK), the packer can write the warp output DIRECTLY
into them via the MPS ``copy_`` path -- the same GPU-side write the
IOSurface stage uses -- and the viewer submits ``CopyBufferToImage``
with no CPU touch of the payload.

Hand-off safety: a source is "pending" from the moment the packer takes
it until the viewer's fence confirms the GPU finished reading it (the
next present's ``vkWaitForFences``). While pending, ``acquire`` returns
None and the packer falls back to the IOSurface stage + memcpy path, so
a slow viewer can never race a warp write.
"""

from __future__ import annotations

import os
import sys
import threading


def direct_staging_enabled() -> bool:
    return (
        sys.platform == "darwin"
        and os.environ.get("D2S_VK_DIRECT_STAGING", "1") not in {"0", "false", "off"}
    )


class DirectSinkRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: list = []

    def register(self, source) -> None:
        with self._lock:
            if source not in self._sources:
                self._sources.append(source)

    def unregister(self, source) -> None:
        with self._lock:
            try:
                self._sources.remove(source)
            except ValueError:
                pass

    def acquire(self, width: int, height: int):
        """Return (source, writable_memoryview) or (None, None)."""
        if not direct_staging_enabled():
            return None, None
        with self._lock:
            for src in self._sources:
                if src.size == (width, height) and src.claim_direct():
                    return src, src.direct_view
        return None, None


DIRECT_SINK = DirectSinkRegistry()
