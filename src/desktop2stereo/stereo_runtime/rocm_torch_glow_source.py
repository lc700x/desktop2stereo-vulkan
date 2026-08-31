"""Torch-computed Glow source for AMD ROCm (no Vulkan compute queue).

Computes the screen-light reduction and the glow texture with torch on the HIP
stream and uploads the RGBA8 texture into a Vulkan image via the working
external-memory image import (``hipMemcpy2DToArray``).

The producer path is NON-BLOCKING: ``submit`` only enqueues the source downsample
(interpolate) and hands the small tensor to a background worker thread. The
worker performs every blocking step (event wait, gather/reduction, screen-light
readback, synchronous HIP copy, and the GENERAL<->SHADER_READ layout
transitions) so the OpenXR frame producer is never stalled — stalling the
producer on the glow dropped the Virtual Desktop session after the first frames.

The math mirrors ``shaders/d2s_glow_source.comp`` (8x8 stratified reduction,
sRGB decode, temporal smoothing) so the visual matches the CUDA/NVIDIA path,
which keeps the Vulkan compute backend unchanged.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any

from viewer.rocm_vulkan_interop import RocmVulkanImageImporter
from viewer.vulkan_resources import VulkanExportableImage

TARGET_WIDTH = 320
TARGET_HEIGHT = 180


class RocmTorchGlowSource:
    """Compute glow + screen light in torch and publish an RGBA8 Vulkan image."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.vk = context.vk
        self.slot_count = 3
        self.importer = RocmVulkanImageImporter()
        self.slots: list[VulkanExportableImage] = []
        for index in range(self.slot_count):
            image = VulkanExportableImage(
                context,
                TARGET_WIDTH,
                TARGET_HEIGHT,
                format=context.vk.VK_FORMAT_R8G8B8A8_UNORM,
                label=f"torch-glow-{index}",
            )
            self.importer.register_slot(image, wait=False, defer=True)
            self.slots.append(image)
        self._ring = 0
        self._lock = threading.Lock()
        self._current_resource: Any = None
        self._screen_light_rgb = (0.18, 0.18, 0.18)
        self._edge_light_rgb = tuple((0.0, 0.0, 0.0) for _ in range(24))
        self._history = None
        self._history_key: tuple[object, ...] | None = None
        self._history_last_submit = 0.0
        self._last_submit_ms = 0.0
        self._serial = 0
        self._reuse_count = 0
        self._closed = False
        self._frame_slots: dict[int, Any] = {}
        self._leases: dict[int, int] = {}
        # Background worker: every blocking glow step runs here, never on the
        # frame producer.
        self._pending: list[tuple[Any, Any, str, float]] = []
        self._pending_condition = threading.Condition()
        self._worker = threading.Thread(
            target=self._worker_loop, name="rocm-torch-glow", daemon=True
        )
        self._worker.start()

    @property
    def capabilities(self):
        return self.importer.capabilities

    @staticmethod
    def _srgb_to_linear(tensor):
        import torch

        low = tensor / 12.92
        high = torch.pow(
            (tensor + 0.055) / 1.055, torch.full_like(tensor, 2.4)
        )
        return torch.where(tensor > 0.04045, high, low)

    @staticmethod
    def _prepare_source(source):
        import torch

        if not isinstance(source, torch.Tensor) or not source.is_cuda:
            return None
        value = source
        if value.ndim == 3:
            if int(value.shape[-1]) in (3, 4):
                value = value[..., :3].permute(2, 0, 1).unsqueeze(0)
            elif int(value.shape[0]) in (3, 4):
                value = value[:3].unsqueeze(0)
        if value.ndim != 4 or int(value.shape[0]) != 1 or int(value.shape[1]) < 3:
            return None
        value = value[:, :3]
        if value.dtype == torch.uint8:
            value = value.to(dtype=torch.float32).div_(255.0)
        elif value.dtype != torch.float32:
            value = value.to(dtype=torch.float32)
        hist, wid = int(value.shape[-2]), int(value.shape[-1])
        longest = max(hist, wid)
        max_side = 256
        env_max_side = os.environ.get("D2S_GLOW_MAX_SIDE")
        if env_max_side:
            try:
                max_side = max(128, int(env_max_side))
            except (TypeError, ValueError):
                pass
        if longest > max_side:
            scale = max_side / float(longest)
            nh = max(1, int(round(hist * scale)))
            nw = max(1, int(round(wid * scale)))
            import torch.nn.functional as functional

            value = functional.interpolate(
                value, size=(nh, nw), mode="bilinear", align_corners=False
            )
        return value.contiguous()

    def _pick_slot(self):
        for _ in range(self.slot_count):
            slot = self.slots[self._ring]
            self._ring = (self._ring + 1) % self.slot_count
            if self._leases.get(id(slot), 0) == 0:
                return slot
        return None

    def submit(
        self,
        source: Any,
        *,
        mode: str = "glow",
        screen_light_only: bool = False,
        temporal_smoothing_seconds: float = 0.10,
    ) -> bool:
        if self._closed:
            return False
        import torch

        start = time.perf_counter()
        value = self._prepare_source(source)
        if value is None:
            return False
        event = torch.cuda.Event()
        event.record()
        with self._pending_condition:
            # Keep only the newest pending sample: the glow is a slow effect and
            # the worker should never lag behind the producer.
            self._pending = [(value, event, str(mode or "").strip().lower(), start)]
            self._pending_condition.notify()
        self._last_submit_ms = (time.perf_counter() - start) * 1000.0
        return True

    def _worker_loop(self) -> None:
        import torch

        while True:
            with self._pending_condition:
                while not self._pending and not self._closed:
                    self._pending_condition.wait()
                if self._closed and not self._pending:
                    return
                value, event, normalized_mode, start = self._pending.pop()
            try:
                event.synchronize()
                rgb = value[0]  # [3,H,W] sRGB-encoded float
                height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
                device = rgb.device
                surround = normalized_mode == "surround"

                # Screen light: 8x8 stratified mean of sRGB->linear.
                sy = (
                    (torch.arange(8, device=device, dtype=torch.float32) + 0.5)
                    / 8.0 * height
                ).floor().long().clamp(0, height - 1)
                sx = (
                    (torch.arange(8, device=device, dtype=torch.float32) + 0.5)
                    / 8.0 * width
                ).floor().long().clamp(0, width - 1)
                grid = rgb[:, sy][:, :, sx]
                screen = self._srgb_to_linear(grid).mean(dim=(1, 2))
                screen_values = tuple(
                    max(0.0, min(8.0, float(v))) for v in screen.tolist()
                )
                with self._lock:
                    self._screen_light_rgb = screen_values

                prefilter_scale = (
                    256.0
                    if normalized_mode in ("glow", "screen", "surround")
                    else 1.0
                )
                out_h, out_w = TARGET_HEIGHT, TARGET_WIDTH
                oy = (
                    torch.arange(out_h, device=device, dtype=torch.float32) + 0.5
                ) / out_h
                ox = (
                    torch.arange(out_w, device=device, dtype=torch.float32) + 0.5
                ) / out_w
                center_x = ox[None, :] * width
                center_y = (1.0 - oy[:, None]) * height
                footprint_x = max(width / out_w, prefilter_scale)
                footprint_y = max(height / out_h, prefilter_scale)
                offx = (
                    torch.arange(8, device=device, dtype=torch.float32) + 0.5
                ) / 8.0 - 0.5
                offy = (
                    torch.arange(8, device=device, dtype=torch.float32) + 0.5
                ) / 8.0 - 0.5
                px = (
                    center_x[None, None, :, :]
                    + offx[:, None, None, None] * footprint_x
                ).floor().long().clamp(0, width - 1)
                py = (
                    center_y[None, None, :, :]
                    + offy[None, :, None, None] * footprint_y
                ).floor().long().clamp(0, height - 1)
                sampled = rgb[:, py, px]
                if surround:
                    total = sampled.mean(dim=(1, 2))
                    total = self._srgb_to_linear(total)
                else:
                    total = self._srgb_to_linear(sampled).mean(dim=(1, 2))

                # Temporal smoothing (matches the shader's history mix).
                history_key = (normalized_mode, width, height)
                with self._lock:
                    history = self._history
                    history_matches = history_key == self._history_key
                    history_last = self._history_last_submit
                if history_matches and history is not None:
                    elapsed = max(0.0, start - history_last)
                    smoothing = max(0.0, float(temporal_smoothing_seconds))
                    if smoothing > 0.0:
                        alpha = 1.0 - math.exp(-elapsed / smoothing)
                        if alpha < 0.9999:
                            total = history * (1.0 - alpha) + total * alpha
                with self._lock:
                    self._history = total.clone()
                    self._history_key = history_key
                    self._history_last_submit = start

                edge_light_values = None
                if surround:
                    edge_light_values = self._compute_edge_lights(rgb, device)

                rgba = (
                    (total.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
                )
                rgba = rgba.permute(1, 2, 0).contiguous()
                alpha_ch = torch.full(
                    (out_h, out_w, 1), 255, dtype=torch.uint8, device=device
                )
                rgba = torch.cat([rgba, alpha_ch], dim=2).contiguous()

                if edge_light_values is not None:
                    with self._lock:
                        self._edge_light_rgb = edge_light_values
                slot = self._pick_slot()
                if slot is None:
                    continue
                state = self.context.image_state(slot.resource.image)
                if (
                    state.layout
                    == self.vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                ):
                    self.context.release_external_image_from_sampling(
                        slot.resource
                    )
                self.importer.copy_tensor(rgba, slot, synchronous=True)
                self.context.prepare_external_image_for_sampling(slot.resource)
                with self._lock:
                    self._current_resource = slot.resource
                    self._serial += 1
                    self._last_submit_ms = (
                        time.perf_counter() - start
                    ) * 1000.0
            except Exception as exc:  # pragma: no cover - defensive
                if os.environ.get("D2S_GLOW_DIAGNOSTIC"):
                    print(
                        "[VulkanOutput] torch glow worker error: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

    def _compute_edge_lights(self, rgb, device):
        height, width = int(rgb.shape[-2]), int(rgb.shape[-1])
        band = max(4.0, min(16.0, round(min(height, width) / 270.0)))
        band = int(band)
        values: list[tuple[float, float, float]] = []
        top = rgb[:, :band, :]
        for index in range(8):
            segment = top[:, :, index * width // 8 : (index + 1) * width // 8]
            values.append(
                tuple(
                    float(v)
                    for v in self._srgb_to_linear(segment.mean(dim=(1, 2))).tolist()
                )
            )
        right = rgb[:, :, width - band :]
        for index in range(4):
            segment = right[:, index * height // 4 : (index + 1) * height // 4, :]
            values.append(
                tuple(
                    float(v)
                    for v in self._srgb_to_linear(segment.mean(dim=(1, 2))).tolist()
                )
            )
        bottom = rgb[:, height - band :, :]
        for index in range(8):
            segment = bottom[:, :, index * width // 8 : (index + 1) * width // 8]
            values.append(
                tuple(
                    float(v)
                    for v in self._srgb_to_linear(segment.mean(dim=(1, 2))).tolist()
                )
            )
        left = rgb[:, :, :band]
        for index in range(4):
            segment = left[:, index * height // 4 : (index + 1) * height // 4, :]
            values.append(
                tuple(
                    float(v)
                    for v in self._srgb_to_linear(segment.mean(dim=(1, 2))).tolist()
                )
            )
        return tuple(
            tuple(max(0.0, min(8.0, float(channel))) for channel in item)
            for item in values
        )

    def poll(self) -> None:
        return

    def acquire(self, frame_id: int) -> dict[str, object]:
        with self._lock:
            screen_light = self._screen_light_rgb
            edge_light = self._edge_light_rgb
            resource = self._current_resource
            serial = self._serial
            submit_ms = self._last_submit_ms
        metadata: dict[str, object] = {
            "screen_light_linear_rgb": screen_light,
            "screen_light_sample_path": "torch_compute_reduction",
            "screen_edge_light_linear_rgb": edge_light,
            "_vulkan_glow_release": self.release_frame,
        }
        if resource is None:
            return metadata
        frame_key = int(frame_id)
        existing = self._frame_slots.get(frame_key)
        if existing is None:
            self._frame_slots[frame_key] = resource
            self._leases[id(resource)] = self._leases.get(id(resource), 0) + 1
        self._reuse_count += 1
        metadata.update(
            {
                "glow_vulkan_image": resource,
                "glow_vulkan_serial": serial,
                "glow_source_path": "torch_compute_external_image",
                "glow_gpu_submit_ms": submit_ms,
                "glow_reuse": self._reuse_count,
            }
        )
        return metadata

    def release_frame(self, frame_id: int) -> None:
        resource = self._frame_slots.pop(int(frame_id), None)
        if resource is None:
            return
        remaining = self._leases.get(id(resource), 1) - 1
        if remaining > 0:
            self._leases[id(resource)] = remaining
        else:
            self._leases.pop(id(resource), None)

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            with self._pending_condition:
                self._pending_condition.notify_all()
            worker = getattr(self, "_worker", None)
            if worker is not None and worker.is_alive():
                worker.join(timeout=2.0)
            importer = getattr(self, "importer", None)
            if importer is not None:
                importer.close()
            for slot in getattr(self, "slots", ()):
                if slot is not None:
                    slot.close()
        finally:
            self.slots = []
            self._frame_slots = {}
            self._current_resource = None
