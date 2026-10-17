"""Registry connecting the packer thread to the Vulkan viewer's staging.

The native macOS path keeps capture, Core ML preprocessing/inference, depth
sanitization, and warp packing on the GPU. MoltenVK does not expose its
mapped transfer allocation as a valid Metal ``MTLBuffer`` on this host, so
the bridge records one explicit GPU-warp-to-Vulkan mapped-buffer handoff
copy. The viewer still submits ``CopyBufferToImage`` from the claimed ring
slot without a CPU color conversion or NumPy packing fallback.

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

    def release(self, source) -> None:
        """Return a claimed but unpublished slot to the available pool."""
        release = getattr(source, "_release_direct", None)
        if callable(release):
            release()

    def pack_native(self, frame, size: tuple[int, int], output_format: str):
        """Pack under the registry lock so unregister/unmap waits for writes."""
        if not direct_staging_enabled():
            return None
        with self._lock:
            for source in self._sources:
                if source.size != size or not source.claim_direct():
                    continue
                try:
                    frame.pack(source.direct_view, size, output_format)
                except Exception:
                    source._release_direct()
                    raise
                return source
        return None


DIRECT_SINK = DirectSinkRegistry()
