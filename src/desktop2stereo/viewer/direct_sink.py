"""Registry connecting the packer thread to the Vulkan viewer's staging.

The native macOS path keeps capture, Core ML preprocessing/inference, depth
sanitization, and warp packing on the GPU. When ``VK_EXT_metal_objects`` can
export the transfer allocation as an ``MTLBuffer``, the warp writes directly
into the Vulkan ring slot. GPU-only mode requires this alias and never uses the
mapped-buffer handoff; the non-strict compatibility path can still use it.

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
        self._metal_alias_verified: bool | None = None
        self._ring_fallback_count = 0

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

    def _pack_mapped_fallback(self, frame, size, output_format, source):
        if source is None or getattr(source, "size", None) != size:
            return None
        destination = getattr(source, "_mapped_view", None)
        if destination is None:
            return None
        frame.pack(destination, size, output_format)
        metrics = getattr(frame, "last_pack_metrics", None)
        if isinstance(metrics, dict):
            metrics["metal_vulkan_ring_fallback"] = True
        self._ring_fallback_count += 1
        if self._ring_fallback_count == 1 or self._ring_fallback_count % 60 == 0:
            print(
                "[DirectSink] Vulkan ring has no free slot; used the mapped-buffer handoff",
                flush=True,
            )
        return source

    def pack_native(
        self,
        frame,
        size: tuple[int, int],
        output_format: str,
        *,
        fallback_source=None,
        gpu_required: bool = False,
    ):
        """Pack under the registry lock so unregister/unmap waits for writes."""
        if not direct_staging_enabled():
            if gpu_required:
                raise RuntimeError(
                    "GPU-only native output requires Vulkan Metal direct staging"
                )
            return self._pack_mapped_fallback(
                frame, size, output_format, fallback_source
            )
        with self._lock:
            for source in self._sources:
                if source.size != size or not source.claim_direct():
                    continue
                try:
                    metal_buffer = getattr(source, "_metal_buffer", None)
                    metal_capacity = int(
                        getattr(source, "_metal_buffer_capacity", 0) or 0
                    )
                    pack_metal = getattr(frame, "pack_metal", None)
                    if gpu_required:
                        if metal_buffer is None or not callable(pack_metal):
                            raise RuntimeError(
                                "GPU-only native output requires an exported Metal buffer"
                            )
                        pack_metal(
                            metal_buffer,
                            metal_capacity,
                            size,
                            output_format,
                        )
                        self._metal_alias_verified = True
                    elif (
                        metal_buffer is not None
                        and callable(pack_metal)
                        and self._metal_alias_verified is not False
                    ):
                        if self._metal_alias_verified is None:
                            expected_size = int(size[0]) * int(size[1]) * 4
                            reference = bytearray(expected_size)
                            frame.pack(memoryview(reference), size, output_format)
                            expected = bytes(reference)
                            try:
                                pack_metal(
                                    metal_buffer,
                                    metal_capacity,
                                    size,
                                    output_format,
                                )
                            except Exception as exc:
                                self._metal_alias_verified = False
                                frame.pack(source.direct_view, size, output_format)
                                print(
                                    "[DirectSink] Metal buffer write failed validation; "
                                    f"using mapped-buffer handoff ({exc})",
                                    flush=True,
                                )
                            else:
                                actual = source.direct_view[:expected_size]
                                if bytes(actual) == expected:
                                    self._metal_alias_verified = True
                                    print(
                                        "[DirectSink] Metal buffer output matches the "
                                        "mapped-buffer reference; direct alias active",
                                        flush=True,
                                    )
                                else:
                                    self._metal_alias_verified = False
                                    frame.pack(
                                        source.direct_view, size, output_format
                                    )
                                    print(
                                        "[DirectSink] Metal buffer output differed from "
                                        "the mapped-buffer reference; using handoff copy",
                                        flush=True,
                                    )
                        elif self._metal_alias_verified:
                            pack_metal(
                                metal_buffer,
                                metal_capacity,
                                size,
                                output_format,
                            )
                        else:
                            frame.pack(source.direct_view, size, output_format)
                    else:
                        frame.pack(source.direct_view, size, output_format)
                except Exception:
                    source._release_direct()
                    raise
                return source
            if gpu_required:
                if any(getattr(source, "size", None) == size for source in self._sources):
                    return None
                raise RuntimeError(
                    f"GPU-only native output has no registered Metal staging slot for {size}"
                )
            return self._pack_mapped_fallback(
                frame, size, output_format, fallback_source
            )


DIRECT_SINK = DirectSinkRegistry()
