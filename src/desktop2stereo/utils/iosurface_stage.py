"""IOSurface-backed SBS staging ring (D2S_SBS_STAGE=1, default on).

A fixed ring of global IOSurfaces serves as the destination for the
device->host DMA of packed SBS frames: torch copies straight into the
surface base via a frombuffer alias, so no intermediate CPU tensor is
allocated per frame (the .cpu() allocation copy disappears).

Lifetime: the packer rotates through ``n`` slots; consumers hold a tuple
referencing one slot's view. With n >= queue depth the oldest slot is not
reused until newer frames have been published (latest-wins drops make
tearing practically impossible; correctness relies on consumer reading
before n subsequent packs).
"""

from __future__ import annotations

import os
import sys
import threading


def _fourcc(text: str) -> int:
    b = text.encode("ascii")
    return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]


def enabled() -> bool:
    # Hard platform gate: IOSurfaces do not exist elsewhere. AMD/NVIDIA/
    # Intel machines keep the exact legacy packing path.
    return sys.platform == "darwin" and os.environ.get(
        "D2S_SBS_STAGE", "1"
    ) in ("1", "true", "on")


class SurfaceStage:
    """One IOSurface plus a reusable writable numpy view of its base."""

    def __init__(self, width: int, height: int):
        import IOSurface

        self.width = int(width)
        self.height = int(height)
        self.nbytes = self.width * self.height * 4
        self.surface = IOSurface.IOSurfaceCreate({
            IOSurface.kIOSurfaceWidth: self.width,
            IOSurface.kIOSurfaceHeight: self.height,
            IOSurface.kIOSurfacePixelFormat: _fourcc("RGBA"),
            IOSurface.kIOSurfaceBytesPerElement: 4,
            IOSurface.kIOSurfaceIsGlobal: True,
        })
        self.surface_id = int(IOSurface.IOSurfaceGetID(self.surface))
        self._view = None

    def writable_view(self) -> memoryview:
        """Base-address view; caller writes exactly nbytes."""
        import IOSurface

        if self._view is None:
            IOSurface.IOSurfaceLock(self.surface, 0, None)
            try:
                base = IOSurface.IOSurfaceGetBaseAddress(self.surface)
                self._view = base.as_buffer(self.nbytes)
            finally:
                IOSurface.IOSurfaceUnlock(self.surface, 0, None)
        return self._view


class StageRing:
    """Round-robin ring of SurfaceStages shared across threads."""

    def __init__(self, width: int, height: int, slots: int = 4):
        import IOSurface

        self.slots = max(2, int(slots))
        self._items = [
            SurfaceStage(width, height) for _ in range(self.slots)
        ]
        self._i = 0
        self._lock = threading.Lock()

    def next_slot(self) -> tuple[int, SurfaceStage]:
        with self._lock:
            slot = self._i % self.slots
            self._i += 1
            return slot, self._items[slot]

    def __len__(self) -> int:
        return self.slots


_RING: StageRing | None = None
_RING_LOCK = threading.Lock()


def acquire_stage(width: int, height: int, slots: int = 4) -> tuple[int, SurfaceStage] | None:
    """Process-wide ring; rebuilt when the packed shape changes so render-size
    changes degrade to legacy packing for one frame instead of forever."""
    global _RING
    if not enabled():
        return None
    width, height = int(width), int(height)
    with _RING_LOCK:
        if _RING is not None and (
            _RING._items[0].width != width or _RING._items[0].height != height
        ):
            _RING = None
        if _RING is None:
            try:
                _RING = StageRing(width, height, slots)
            except Exception:
                _RING = None
                return None
    return _RING.next_slot()
