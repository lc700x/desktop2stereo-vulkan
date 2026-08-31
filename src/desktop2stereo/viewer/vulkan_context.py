from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Iterable


class VulkanUnavailableError(RuntimeError):
    pass


class VulkanCapabilityError(RuntimeError):
    pass


def is_vulkan_device_lost_error(exc: BaseException) -> bool:
    """Return whether a Vulkan binding exception reports a lost device."""
    marker = f"{type(exc).__name__} {exc}".lower().replace("_", " ")
    return "devicelost" in "".join(marker.split())


def make_vulkan_version(major: int, minor: int, patch: int = 0) -> int:
    return (int(major) << 22) | (int(minor) << 12) | int(patch)


MIN_VULKAN_API_VERSION = make_vulkan_version(1, 2, 0)


def unpack_vulkan_version(version: int) -> tuple[int, int, int]:
    value = int(version)
    return value >> 22, (value >> 12) & 0x3FF, value & 0xFFF


def format_vulkan_version(version: int) -> str:
    return ".".join(str(part) for part in unpack_vulkan_version(version))


@dataclass(frozen=True, slots=True)
class VulkanDeviceInfo:
    name: str
    api_version: int
    driver_version: int
    vendor_id: int
    device_id: int
    device_type: int
    queue_family_index: int
    compute_queue_family_index: int = -1
    transfer_queue_family_index: int = -1
    timeline_semaphore_enabled: bool = False
    synchronization2_enabled: bool = False
    # Vulkan does not expose a portable DXGI LUID in the base properties
    # query. Keep it explicit so cross-API consumers never infer identity
    # from a name or PCI vendor alone.
    adapter_luid: int = 0

    @property
    def adapter_identity(self) -> str:
        if self.adapter_luid:
            return f"luid:{self.adapter_luid:016x}"
        return f"pci:{self.vendor_id:04x}:{self.device_id:04x}"

    @property
    def api_version_text(self) -> str:
        return format_vulkan_version(self.api_version)


@dataclass(frozen=True, slots=True)
class VulkanContextConfig:
    application_name: str = "Desktop2Stereo Vulkan"
    engine_name: str = "D2S"
    api_version: int = make_vulkan_version(1, 4, 0)
    enable_validation: bool = False
    required_instance_extensions: tuple[str, ...] = ()
    required_device_extensions: tuple[str, ...] = ()
    # Keep in-flight command resources bounded and configurable for validation.
    frame_context_count: int = 3


@dataclass(frozen=True, slots=True)
class QueueFamilySelection:
    graphics: int
    compute: int
    transfer: int


@dataclass(slots=True)
class VulkanFrameContext:
    command_pool: Any
    command_buffer: Any
    fence: Any
    timeline_value: int = 0
    queue_resources: dict[str, "VulkanQueueFrameResources"] = field(default_factory=dict)


@dataclass(slots=True)
class VulkanQueueFrameResources:
    command_pool: Any
    command_buffer: Any
    fence: Any
    timeline_value: int = 0


@dataclass(frozen=True, slots=True)
class ImageState:
    layout: int
    access_mask: int
    stage_mask: int
    queue_family_index: int


@dataclass(frozen=True, slots=True)
class QueueOwnershipTransfer:
    image_key: int
    source_queue_family_index: int
    destination_queue_family_index: int
    state: ImageState


class ImageStateTracker:
    """Tracks the last declared image state for explicit barrier construction."""

    def __init__(self, *, default_queue_family_index: int) -> None:
        self.default_queue_family_index = int(default_queue_family_index)
        self._states: dict[int, ImageState] = {}
        self._pending_transfers: dict[int, QueueOwnershipTransfer] = {}

    def get(self, image_key: int, *, undefined_layout: int) -> ImageState:
        return self._states.get(
            int(image_key),
            ImageState(
                layout=int(undefined_layout),
                access_mask=0,
                stage_mask=0,
                queue_family_index=self.default_queue_family_index,
            ),
        )

    def update(self, image_key: int, state: ImageState) -> None:
        self._states[int(image_key)] = state

    def remove(self, image_key: int) -> None:
        key = int(image_key)
        if key in self._pending_transfers:
            raise VulkanCapabilityError(
                f"Image {key} cannot be removed during a pending queue ownership transfer"
            )
        self._states.pop(key, None)

    def require_owner(self, image_key: int, queue_family_index: int) -> ImageState:
        if int(image_key) in self._pending_transfers:
            raise VulkanCapabilityError(
                f"Image {int(image_key)} has a pending queue ownership transfer"
            )
        state = self._states.get(int(image_key))
        if state is not None and state.queue_family_index not in (
            int(queue_family_index),
        ):
            raise VulkanCapabilityError(
                f"Image {int(image_key)} is owned by queue family "
                f"{state.queue_family_index}, not {int(queue_family_index)}"
            )
        return state

    def begin_ownership_transfer(
        self,
        image_key: int,
        *,
        source_queue_family_index: int,
        destination_queue_family_index: int,
        undefined_layout: int,
    ) -> QueueOwnershipTransfer:
        key = int(image_key)
        if key in self._pending_transfers:
            raise VulkanCapabilityError(f"Image {key} already has a pending ownership transfer")
        state = self.get(key, undefined_layout=undefined_layout)
        if state.queue_family_index != int(source_queue_family_index):
            raise VulkanCapabilityError(
                f"Image {key} is owned by queue family {state.queue_family_index}, "
                f"not {int(source_queue_family_index)}"
            )
        transfer = QueueOwnershipTransfer(
            image_key=key,
            source_queue_family_index=int(source_queue_family_index),
            destination_queue_family_index=int(destination_queue_family_index),
            state=state,
        )
        self._pending_transfers[key] = transfer
        return transfer

    def complete_ownership_transfer(self, transfer: QueueOwnershipTransfer) -> None:
        current = self._pending_transfers.get(int(transfer.image_key))
        if current != transfer:
            raise VulkanCapabilityError("queue ownership transfer does not match tracker state")
        state = transfer.state
        self._states[int(transfer.image_key)] = ImageState(
            layout=state.layout,
            access_mask=state.access_mask,
            stage_mask=state.stage_mask,
            queue_family_index=transfer.destination_queue_family_index,
        )
        del self._pending_transfers[int(transfer.image_key)]

    def clear(self) -> None:
        self._states.clear()
        self._pending_transfers.clear()


class VulkanContext:
    def __init__(
        self,
        *,
        vk: Any,
        instance: Any,
        physical_device: Any,
        device: Any,
        queue: Any,
        queue_family_index: int,
        device_info: VulkanDeviceInfo,
        owns_instance: bool,
        owns_device: bool,
        compute_queue: Any | None = None,
        transfer_queue: Any | None = None,
        compute_queue_family_index: int | None = None,
        transfer_queue_family_index: int | None = None,
        compute_queue_index: int = 0,
        transfer_queue_index: int = 0,
        frame_context_count: int = 3,
    ) -> None:
        self.vk = vk
        self.instance = instance
        self.physical_device = physical_device
        self.device = device
        self.queue = queue
        self.queue_family_index = int(queue_family_index)
        self.graphics_queue = queue
        self.compute_queue = compute_queue if compute_queue is not None else queue
        self.transfer_queue = transfer_queue if transfer_queue is not None else queue
        self.compute_queue_index = int(compute_queue_index)
        self.transfer_queue_index = int(transfer_queue_index)
        self.compute_queue_family_index = int(
            compute_queue_family_index
            if compute_queue_family_index is not None
            else queue_family_index
        )
        self.transfer_queue_family_index = int(
            transfer_queue_family_index
            if transfer_queue_family_index is not None
            else queue_family_index
        )
        self.device_info = device_info
        if int(frame_context_count) < 1:
            raise ValueError("frame_context_count must be at least one")
        self.frame_context_count = int(frame_context_count)
        self._owns_instance = bool(owns_instance)
        self._owns_device = bool(owns_device)
        self._command_pool = None
        self._command_buffer = None
        self._fence = None
        self._frame_contexts: list[VulkanFrameContext] = []
        self._frame_indices = {role: 0 for role in ("graphics", "compute", "transfer")}
        self._timeline_semaphore = None
        self._timeline_value = 0
        self._image_states = ImageStateTracker(
            default_queue_family_index=self.queue_family_index
        )
        self._lock = RLock()
        self._closed = False
        self._device_lost = False
        self._device_lost_error: str | None = None
        self._create_command_resources()
        if self.device_info.timeline_semaphore_enabled:
            self._timeline_semaphore = _create_timeline_semaphore(vk, self.device)

    def get_queue(self, role: str = "graphics") -> Any:
        queues = {
            "graphics": self.graphics_queue,
            "compute": self.compute_queue,
            "transfer": self.transfer_queue,
        }
        try:
            return queues[str(role).lower()]
        except KeyError as exc:
            raise ValueError(f"unknown Vulkan queue role: {role}") from exc

    def queue_family(self, role: str = "graphics") -> int:
        families = {
            "graphics": self.queue_family_index,
            "compute": self.compute_queue_family_index,
            "transfer": self.transfer_queue_family_index,
        }
        try:
            return families[str(role).lower()]
        except KeyError as exc:
            raise ValueError(f"unknown Vulkan queue role: {role}") from exc

    @classmethod
    def create(cls, config: VulkanContextConfig | None = None) -> "VulkanContext":
        cfg = config or VulkanContextConfig()
        vk = _import_vulkan()
        available_layers = _enumerate_names(vk.vkEnumerateInstanceLayerProperties(), "layerName")
        available_extensions = _enumerate_names(
            vk.vkEnumerateInstanceExtensionProperties(None),
            "extensionName",
        )
        required_instance_extensions = _require_names(
            "Vulkan instance extension",
            cfg.required_instance_extensions,
            available_extensions,
        )
        layers: tuple[str, ...] = ()
        if cfg.enable_validation:
            validation_layer = "VK_LAYER_KHRONOS_validation"
            if validation_layer not in available_layers:
                raise VulkanCapabilityError(
                    f"Vulkan validation requested but {validation_layer} is unavailable"
                )
            layers = (validation_layer,)

        loader_api_version = _loader_api_version(vk)
        instance_api_version = min(int(cfg.api_version), loader_api_version)
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName=cfg.application_name,
            applicationVersion=1,
            pEngineName=cfg.engine_name,
            engineVersion=1,
            apiVersion=instance_api_version,
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
            enabledLayerCount=len(layers),
            ppEnabledLayerNames=list(layers) or None,
            enabledExtensionCount=len(required_instance_extensions),
            ppEnabledExtensionNames=list(required_instance_extensions) or None,
        )

        instance = None
        device = None
        try:
            instance = vk.vkCreateInstance(create_info, None)
            physical_device, queue_families = _select_physical_device(vk, instance)
            available_device_extensions = _enumerate_names(
                vk.vkEnumerateDeviceExtensionProperties(physical_device, None),
                "extensionName",
            )
            required_device_extensions = _require_names(
                "Vulkan device extension",
                cfg.required_device_extensions,
                available_device_extensions,
            )
            device, synchronization2_enabled = _create_device(
                vk,
                physical_device,
                queue_families,
                required_device_extensions,
            )
            queue = vk.vkGetDeviceQueue(device, queue_families.graphics, 0)
            compute_queue = vk.vkGetDeviceQueue(device, queue_families.compute, 0)
            transfer_queue = vk.vkGetDeviceQueue(device, queue_families.transfer, 0)
            info = _device_info(
                vk,
                physical_device,
                queue_families,
                timeline_semaphore_enabled=True,
                synchronization2_enabled=synchronization2_enabled,
            )
            return cls(
                vk=vk,
                instance=instance,
                physical_device=physical_device,
                device=device,
                queue=queue,
                queue_family_index=queue_families.graphics,
                device_info=info,
                owns_instance=True,
                owns_device=True,
                compute_queue=compute_queue,
                transfer_queue=transfer_queue,
                compute_queue_family_index=queue_families.compute,
                transfer_queue_family_index=queue_families.transfer,
                frame_context_count=cfg.frame_context_count,
            )
        except Exception:
            if device is not None:
                vk.vkDestroyDevice(device, None)
            if instance is not None:
                vk.vkDestroyInstance(instance, None)
            raise

    @classmethod
    def adopt(
        cls,
        *,
        instance: Any,
        physical_device: Any,
        device: Any,
        queue_family_index: int,
        owns_instance: bool = True,
        owns_device: bool = True,
        timeline_semaphore_enabled: bool = False,
        synchronization2_enabled: bool = False,
        compute_queue_index: int = 0,
        transfer_queue_index: int = 0,
        frame_context_count: int = 3,
    ) -> "VulkanContext":
        vk = _import_vulkan()
        queue = vk.vkGetDeviceQueue(device, int(queue_family_index), 0)
        compute_queue = vk.vkGetDeviceQueue(
            device, int(queue_family_index), int(compute_queue_index)
        )
        transfer_queue = vk.vkGetDeviceQueue(
            device, int(queue_family_index), int(transfer_queue_index)
        )
        return cls(
            vk=vk,
            instance=instance,
            physical_device=physical_device,
            device=device,
            queue=queue,
            queue_family_index=int(queue_family_index),
            device_info=_device_info(
                vk,
                physical_device,
                QueueFamilySelection(
                    graphics=int(queue_family_index),
                    compute=int(queue_family_index),
                    transfer=int(queue_family_index),
                ),
                timeline_semaphore_enabled=timeline_semaphore_enabled,
                synchronization2_enabled=synchronization2_enabled,
            ),
            owns_instance=owns_instance,
            owns_device=owns_device,
            compute_queue=compute_queue,
            transfer_queue=transfer_queue,
            compute_queue_family_index=int(queue_family_index),
            transfer_queue_family_index=int(queue_family_index),
            compute_queue_index=int(compute_queue_index),
            transfer_queue_index=int(transfer_queue_index),
            frame_context_count=int(frame_context_count),
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def command_pool(self) -> Any:
        self._ensure_open()
        return self._command_pool

    @property
    def last_submitted_timeline_value(self) -> int:
        self._ensure_open()
        return self._timeline_value

    def completed_timeline_value(self) -> int | None:
        """Return the completed timeline value without waiting for GPU work."""
        with self._lock:
            self._ensure_open()
            if self._timeline_semaphore is None:
                return None
            get_value = getattr(self.vk, "vkGetSemaphoreCounterValue", None)
            if get_value is None:
                return None
            try:
                return int(get_value(self.device, self._timeline_semaphore))
            except Exception:
                return None

    def image_handle_from_address(self, address: int) -> Any:
        self._ensure_open()
        if not address:
            raise ValueError("VkImage address must be non-zero")
        return self.vk.ffi.cast("VkImage", int(address))

    def register_image_state(self, image: Any, state: ImageState) -> None:
        self._ensure_open()
        self._image_states.update(_cffi_handle_address(self.vk, image), state)

    def register_external_image(self, resource: Any) -> None:
        if getattr(resource, "context", None) is not self:
            raise VulkanCapabilityError(
                "external image belongs to a different Vulkan context"
            )
        from viewer.vulkan_resources import VulkanExternalImageRegistry

        registry = getattr(self, "_external_image_registry", None)
        if registry is None:
            registry = VulkanExternalImageRegistry(self)
            self._external_image_registry = registry
        registry.register(resource)

    def adopt_external_depth_attachment(self, resource: Any) -> None:
        """Adopt the producer's completed depth layout for a later consumer.

        The producer completion semaphore is still required by the consumer
        submission. This method only records the layout/access contract after
        that semaphore has been published; it does not submit a barrier or
        claim that the image is already safe to sample.
        """
        self._ensure_open()
        if getattr(resource, "context", self) is not self:
            raise VulkanCapabilityError(
                "external depth image belongs to a different context"
            )
        vk = self.vk
        image_key = _cffi_handle_address(vk, resource.image)
        self._image_states.require_owner(image_key, self.queue_family_index)
        self._image_states.update(
            image_key,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                access_mask=vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                stage_mask=(
                    vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                    | vk.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                ),
                queue_family_index=self.queue_family_index,
            ),
        )

    def prepare_external_depth_for_sampling(
        self,
        resource: Any,
        *,
        wait_for_timeline: int | None = None,
        wait_semaphore: Any | None = None,
        wait_semaphore_value: int | None = None,
        signal_semaphore: Any | None = None,
        signal_semaphore_value: int | None = None,
    ) -> int:
        """Transition an adopted Filament depth image to shader-read layout."""
        self._ensure_open()
        if getattr(resource, "context", self) is not self:
            raise VulkanCapabilityError(
                "external depth image belongs to a different context"
            )
        vk = self.vk
        image_key = _cffi_handle_address(vk, resource.image)
        state = self._image_states.get(
            image_key, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self._image_states.require_owner(image_key, self.queue_family_index)
        if state.layout != vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            raise VulkanCapabilityError(
                "external depth image must be adopted from the attachment layout"
            )
        aspect_mask = int(
            getattr(resource, "_aspect_mask", vk.VK_IMAGE_ASPECT_DEPTH_BIT)
        )

        def record(command_buffer: Any) -> None:
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=state.access_mask,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=state.layout,
                newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=resource.image,
                subresourceRange=_depth_subresource_range(
                    vk, aspect_mask=aspect_mask
                ),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                state.stage_mask,
                vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        timeline_value = self.submit_on(
            "graphics",
            record,
            wait_for_timeline=wait_for_timeline,
            wait_semaphore=wait_semaphore,
            wait_semaphore_value=wait_semaphore_value,
            signal_semaphore=signal_semaphore,
            signal_semaphore_value=signal_semaphore_value,
        )
        self._image_states.update(
            image_key,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                queue_family_index=self.queue_family_index,
            ),
        )
        return int(timeline_value)

    def release_external_depth_from_sampling(
        self,
        resource: Any,
        *,
        wait_for_timeline: int | None = None,
    ) -> int:
        """Return a sampled depth image to Filament's attachment layout."""
        self._ensure_open()
        if getattr(resource, "context", self) is not self:
            raise VulkanCapabilityError(
                "external depth image belongs to a different context"
            )
        vk = self.vk
        image_key = _cffi_handle_address(vk, resource.image)
        state = self._image_states.get(
            image_key, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self._image_states.require_owner(image_key, self.queue_family_index)
        if state.layout != vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            raise VulkanCapabilityError(
                "external depth image must be shader-readable before release"
            )
        aspect_mask = int(
            getattr(resource, "_aspect_mask", vk.VK_IMAGE_ASPECT_DEPTH_BIT)
        )

        def record(command_buffer: Any) -> None:
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                dstAccessMask=vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                oldLayout=state.layout,
                newLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=resource.image,
                subresourceRange=_depth_subresource_range(
                    vk, aspect_mask=aspect_mask
                ),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                | vk.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        timeline_value = self.submit_on(
            "graphics", record, wait_for_timeline=wait_for_timeline
        )
        self._image_states.update(
            image_key,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                access_mask=vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                stage_mask=(
                    vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                    | vk.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
                ),
                queue_family_index=self.queue_family_index,
            ),
        )
        return int(timeline_value)

    def unregister_external_image(self, resource: Any) -> None:
        registry = getattr(self, "_external_image_registry", None)
        if registry is None:
            raise VulkanCapabilityError("external image registry is not initialized")
        registry.unregister(resource)

    def unregister_image_state(self, image: Any) -> None:
        self._ensure_open()
        self._image_states.remove(_cffi_handle_address(self.vk, image))

    def image_state(self, image: Any) -> ImageState:
        self._ensure_open()
        return self._image_states.get(
            _cffi_handle_address(self.vk, image),
            undefined_layout=self.vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )

    def clear_color_image(
        self,
        image: Any,
        color: tuple[float, float, float, float],
        *,
        base_array_layer: int = 0,
        layer_count: int = 1,
        wait_semaphore: Any | None = None,
        wait_semaphore_value: int | None = None,
    ) -> int:
        if len(color) != 4:
            raise ValueError("clear color must contain four components")
        if int(base_array_layer) < 0 or int(layer_count) < 1:
            raise ValueError("clear image array range is invalid")
        vk = self.vk
        image_key = _cffi_handle_address(vk, image)
        old_state = self._image_states.get(
            image_key,
            undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self._image_states.require_owner(image_key, self.queue_family_index)
        old_layout = old_state.layout

        def record(command_buffer: Any) -> None:
            source_access = old_state.access_mask
            source_stage = old_state.stage_mask or vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            to_transfer = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=source_access,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout=old_layout,
                newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=image,
                subresourceRange=_color_subresource_range(
                    vk,
                    base_array_layer=base_array_layer,
                    layer_count=layer_count,
                ),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                source_stage,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [to_transfer],
            )
            vk.vkCmdClearColorImage(
                command_buffer,
                image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                vk.VkClearColorValue(float32=list(float(component) for component in color)),
                1,
                [
                    _color_subresource_range(
                        vk,
                        base_array_layer=base_array_layer,
                        layer_count=layer_count,
                    )
                ],
            )
            to_runtime = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                newLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=image,
                subresourceRange=_color_subresource_range(
                    vk,
                    base_array_layer=base_array_layer,
                    layer_count=layer_count,
                ),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [to_runtime],
            )

        timeline_value = self.submit_on(
            "graphics",
            record,
            wait_semaphore=wait_semaphore,
            wait_semaphore_value=wait_semaphore_value,
        )
        self._image_states.update(
            image_key,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                access_mask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                stage_mask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                queue_family_index=self.queue_family_index,
            ),
        )

        return timeline_value

    def compose_sbs_images(
        self,
        left_resource: Any,
        right_resource: Any,
        destination_resource: Any,
        *,
        wait_for_timeline: int | None = None,
    ) -> int:
        """Blit two Vulkan eye images into a D3D11-imported SBS image.

        This is the compatibility bridge for the current two-eye Vulkan
        producer. It performs GPU-only composition and records two GPU blits;
        it deliberately reports a non-strict zero-copy boundary to callers.
        A later fused shader can write the imported SBS image directly.
        """
        self._ensure_open()
        resources = (left_resource, right_resource, destination_resource)
        for resource in resources:
            if getattr(resource, "context", self) is not self:
                raise VulkanCapabilityError("SBS composition resource belongs to a different context")
        left_width = int(getattr(left_resource, "width", 0))
        left_height = int(getattr(left_resource, "height", 0))
        right_width = int(getattr(right_resource, "width", 0))
        right_height = int(getattr(right_resource, "height", 0))
        destination_width = int(getattr(destination_resource, "width", 0))
        destination_height = int(getattr(destination_resource, "height", 0))
        if min(left_width, left_height, right_width, right_height) < 1:
            raise ValueError("SBS source image dimensions must be positive")
        if (right_width, right_height) != (left_width, left_height):
            raise ValueError("SBS source eye dimensions must match")
        if (destination_width, destination_height) != (left_width * 2, left_height):
            raise ValueError("D3D11 SBS destination dimensions must be two eye widths")
        vk = self.vk
        blit_src = int(getattr(vk, "VK_FORMAT_FEATURE_BLIT_SRC_BIT", 0))
        blit_dst = int(getattr(vk, "VK_FORMAT_FEATURE_BLIT_DST_BIT", 0))
        if blit_src and blit_dst:
            for resource, required in (
                (left_resource, blit_src),
                (right_resource, blit_src),
                (destination_resource, blit_dst),
            ):
                properties = vk.vkGetPhysicalDeviceFormatProperties(
                    self.physical_device, int(resource.format)
                )
                if not int(properties.optimalTilingFeatures) & required:
                    raise VulkanCapabilityError(
                        f"Vulkan format {int(resource.format)} lacks required SBS blit support"
                    )
        image_states = []
        for resource in resources:
            image_key = _cffi_handle_address(vk, resource.image)
            state = self._image_states.get(
                image_key, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
            )
            self._image_states.require_owner(image_key, self.queue_family_index)
            image_states.append((resource, image_key, state))

        def record(command_buffer: Any) -> None:
            for resource, _image_key, state in image_states:
                barrier = vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=int(state.access_mask),
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT
                    if resource is not destination_resource
                    else vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=state.layout,
                    newLayout=(
                        vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                        if resource is not destination_resource
                        else vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                    ),
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=resource.image,
                    subresourceRange=_color_subresource_range(vk),
                )
                vk.vkCmdPipelineBarrier(
                    command_buffer,
                    int(state.stage_mask) or vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0,
                    None,
                    0,
                    None,
                    1,
                    [barrier],
                )

            def region(source: Any, destination_x: int) -> Any:
                return vk.VkImageBlit(
                    srcSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                    srcOffsets=[
                        vk.VkOffset3D(x=0, y=0, z=0),
                        vk.VkOffset3D(x=int(source.width), y=int(source.height), z=1),
                    ],
                    dstSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                    dstOffsets=[
                        vk.VkOffset3D(x=destination_x, y=0, z=0),
                        vk.VkOffset3D(
                            x=destination_x + int(source.width),
                            y=int(source.height),
                            z=1,
                        ),
                    ],
                )

            vk.vkCmdBlitImage(
                command_buffer,
                left_resource.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                destination_resource.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                [region(left_resource, 0)],
                vk.VK_FILTER_NEAREST,
            )
            vk.vkCmdBlitImage(
                command_buffer,
                right_resource.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                destination_resource.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                [region(right_resource, left_width)],
                vk.VK_FILTER_NEAREST,
            )

            for resource, _image_key, state in image_states:
                if resource is destination_resource:
                    old_layout = vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                    old_access = vk.VK_ACCESS_TRANSFER_WRITE_BIT
                    old_stage = vk.VK_PIPELINE_STAGE_TRANSFER_BIT
                    new_access = vk.VK_ACCESS_MEMORY_READ_BIT | vk.VK_ACCESS_MEMORY_WRITE_BIT
                else:
                    old_layout = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                    old_access = vk.VK_ACCESS_TRANSFER_READ_BIT
                    old_stage = vk.VK_PIPELINE_STAGE_TRANSFER_BIT
                    new_access = vk.VK_ACCESS_MEMORY_READ_BIT | vk.VK_ACCESS_MEMORY_WRITE_BIT
                barrier = vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=old_access,
                    dstAccessMask=new_access,
                    oldLayout=old_layout,
                    newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=resource.image,
                    subresourceRange=_color_subresource_range(vk),
                )
                vk.vkCmdPipelineBarrier(
                    command_buffer,
                    old_stage,
                    vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0,
                    0,
                    None,
                    0,
                    None,
                    1,
                    [barrier],
                )

        timeline = self.submit_on(
            "graphics",
            record,
            wait_for_timeline=wait_for_timeline,
        )
        for resource, image_key, _state in image_states:
            self._image_states.update(
                image_key,
                ImageState(
                    layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    access_mask=vk.VK_ACCESS_MEMORY_READ_BIT | vk.VK_ACCESS_MEMORY_WRITE_BIT,
                    stage_mask=vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    queue_family_index=self.queue_family_index,
                ),
            )
        return int(timeline)

    def prepare_external_image_for_cuda(
        self, resource: Any, *, wait: bool = True, defer: bool = False
    ) -> int:
        """Establish a persistent GENERAL layout before CUDA writes external memory.

        ``wait=False`` records the layout transition but does not call
        ``vkDeviceWaitIdle``; callers that register producer images while the
        compositor is actively rendering use it to avoid stalling the whole
        device mid-stream.

        ``defer=True`` skips the GPU barrier submission entirely and only marks
        the image GENERAL in the state tracker. Use it for storage-image sources
        whose first producer access is a HIP copy + compute-read with its own
        barrier.
        """

        self._ensure_open()
        if getattr(resource, "context", self) is not self:
            raise VulkanCapabilityError("external image belongs to a different context")
        image_key = _cffi_handle_address(self.vk, resource.image)
        state = self._image_states.get(
            image_key, undefined_layout=self.vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self._image_states.require_owner(image_key, self.queue_family_index)
        if state.layout == self.vk.VK_IMAGE_LAYOUT_GENERAL:
            return self._timeline_value
        if state.layout != self.vk.VK_IMAGE_LAYOUT_UNDEFINED:
            raise VulkanCapabilityError(
                "CUDA external image must be UNDEFINED or GENERAL during slot registration"
            )
        vk = self.vk
        if defer:
            self._image_states.update(
                image_key,
                ImageState(
                    layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    access_mask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                    stage_mask=vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    queue_family_index=self.queue_family_index,
                ),
            )
            return self._timeline_value

        def record(command_buffer: Any) -> None:
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=0,
                dstAccessMask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=resource.image,
                subresourceRange=_color_subresource_range(vk),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        timeline_value = self.submit_on("graphics", record)
        self._image_states.update(
            image_key,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                access_mask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                stage_mask=vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                queue_family_index=self.queue_family_index,
            ),
        )
        # This is slot initialization only. Runtime frames synchronize through
        # the CUDA stream and never call vkDeviceWaitIdle.
        if wait:
            self.wait_idle()
        return timeline_value

    def prepare_external_image_for_producer(
        self, resource: Any, *, wait: bool = True, defer: bool = False
    ) -> int:
        """Prepare an exportable image for any GPU producer backend."""
        return self.prepare_external_image_for_cuda(resource, wait=wait, defer=defer)

    def prepare_external_image_for_sampling(
        self,
        resource: Any,
        *,
        wait_for_timeline: int | None = None,
        wait_semaphore: Any | None = None,
        wait_semaphore_value: int | None = None,
        signal_semaphore: Any | None = None,
        signal_semaphore_value: int | None = None,
    ) -> int:
        """Transition a producer image to shader-read layout on graphics queue.

        ``wait_semaphore`` is the producer-ready point. The optional
        ``signal_semaphore`` is the post-barrier point consumed by Filament.
        """
        self._ensure_open()
        if getattr(resource, "context", self) is not self:
            raise VulkanCapabilityError("external image belongs to a different context")
        vk = self.vk
        image_key = _cffi_handle_address(vk, resource.image)
        state = self._image_states.get(
            image_key, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self._image_states.require_owner(image_key, self.queue_family_index)
        if state.layout != vk.VK_IMAGE_LAYOUT_GENERAL:
            raise VulkanCapabilityError(
                "Filament source image must be in GENERAL before sampling transition"
            )

        def record(command_buffer: Any) -> None:
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=state.access_mask or vk.VK_ACCESS_MEMORY_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=resource.image,
                subresourceRange=_color_subresource_range(vk),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                state.stage_mask or vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        timeline_value = self.submit_on(
            "graphics",
            record,
            wait_for_timeline=wait_for_timeline,
            wait_semaphore=wait_semaphore,
            wait_semaphore_value=wait_semaphore_value,
            signal_semaphore=signal_semaphore,
            signal_semaphore_value=signal_semaphore_value,
        )
        self._image_states.update(
            image_key,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                access_mask=vk.VK_ACCESS_SHADER_READ_BIT,
                stage_mask=(
                    vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                    | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                ),
                queue_family_index=self.queue_family_index,
            ),
        )
        return timeline_value

    def release_external_image_from_sampling(
        self,
        resource: Any,
        *,
        wait_for_timeline: int | None = None,
        wait_semaphore: Any | None = None,
        wait_semaphore_value: int | None = None,
        signal_semaphore: Any | None = None,
        signal_semaphore_value: int | None = None,
    ) -> int:
        """Return a Filament source image to producer-writable GENERAL layout."""
        # Device loss is terminal.  Cleanup must only release CPU-side leases;
        # submitting another barrier or querying another fence merely creates a
        # cascade of secondary exceptions on an already dead device.
        if self._device_lost:
            return 0
        self._ensure_open()
        if getattr(resource, "context", self) is not self:
            raise VulkanCapabilityError("external image belongs to a different context")
        vk = self.vk
        release_role = (
            "transfer"
            if self.queue_family("transfer") == self.queue_family_index
            else "graphics"
        )
        image_key = _cffi_handle_address(vk, resource.image)
        state = self._image_states.get(
            image_key, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self._image_states.require_owner(image_key, self.queue_family(release_role))
        if state.layout != vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            raise VulkanCapabilityError(
                "Filament source image must be shader-read before consumer release"
            )

        def record(command_buffer: Any) -> None:
            barrier = vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                dstAccessMask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=resource.image,
                subresourceRange=_color_subresource_range(vk),
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        timeline_value = self.submit_on(
            release_role,
            record,
            wait_for_timeline=wait_for_timeline,
            wait_semaphore=wait_semaphore,
            wait_semaphore_value=wait_semaphore_value,
            signal_semaphore=signal_semaphore,
            signal_semaphore_value=signal_semaphore_value,
        )
        self._image_states.update(
            image_key,
            ImageState(
                layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                access_mask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                stage_mask=vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                queue_family_index=self.queue_family(release_role),
            ),
        )
        return timeline_value

    def copy_image(
        self,
        source: Any,
        destination: Any,
        *,
        wait_for_timeline: int | None = None,
        wait_semaphore: Any | None = None,
        wait_semaphore_value: int | None = None,
        flip_x: bool = False,
        flip_y: bool = False,
        filter_mode: int | None = None,
        resize: bool = False,
        source_array_layer: int = 0,
        destination_array_layer: int = 0,
        source_rect: tuple[int, int, int, int] | None = None,
        destination_rect: tuple[int, int, int, int] | None = None,
        destination_host_readable: bool = False,
    ) -> int:
        """Copy one registered Vulkan image layer with optional scaling."""

        self._ensure_open()
        for resource in (source, destination):
            if getattr(resource, "context", self) is not self:
                raise VulkanCapabilityError("Vulkan image belongs to a different context")
        if source.image is destination.image:
            raise VulkanCapabilityError("source and destination Vulkan images must differ")
        dimensions_match = (
            int(source.width) == int(destination.width)
            and int(source.height) == int(destination.height)
        )
        if not dimensions_match and not resize:
            raise ValueError("Vulkan image copy dimensions must match")
        if int(source_array_layer) < 0 or int(destination_array_layer) < 0:
            raise ValueError("image array layers must not be negative")
        if destination_rect is None:
            destination_x0, destination_y0 = 0, 0
            destination_x1, destination_y1 = int(destination.width), int(destination.height)
        else:
            if len(destination_rect) != 4:
                raise ValueError("destination_rect must contain four values")
            destination_x0, destination_y0, destination_x1, destination_y1 = (
                int(value) for value in destination_rect
            )
            if not (
                0 <= destination_x0 < destination_x1 <= int(destination.width)
                and 0 <= destination_y0 < destination_y1 <= int(destination.height)
            ):
                raise ValueError("destination_rect must be inside the destination image")
        if source_rect is None:
            source_x0, source_y0 = 0, 0
            source_x1, source_y1 = int(source.width), int(source.height)
        else:
            if len(source_rect) != 4:
                raise ValueError("source_rect must contain four values")
            source_x0, source_y0, source_x1, source_y1 = (
                int(value) for value in source_rect
            )
            if not (
                0 <= source_x0 < source_x1 <= int(source.width)
                and 0 <= source_y0 < source_y1 <= int(source.height)
            ):
                raise ValueError("source_rect must be inside the source image")

        vk = self.vk
        source_format = int(source.format)
        destination_format = int(destination.format)
        format_pair = frozenset((source_format, destination_format))
        srgb_unorm_pairs = {
            frozenset((
                int(vk.VK_FORMAT_R8G8B8A8_UNORM),
                int(vk.VK_FORMAT_R8G8B8A8_SRGB),
            )),
            frozenset((
                int(vk.VK_FORMAT_B8G8R8A8_UNORM),
                int(vk.VK_FORMAT_B8G8R8A8_SRGB),
            )),
        }
        formats_match = source_format == destination_format
        formats_are_srgb_compatible = format_pair in srgb_unorm_pairs
        if not formats_match and not formats_are_srgb_compatible:
            raise ValueError("Vulkan image copy formats must match or be sRGB/UNORM compatible")

        source_key = _cffi_handle_address(vk, source.image)
        destination_key = _cffi_handle_address(vk, destination.image)
        source_state = self._image_states.get(
            source_key, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        destination_state = self._image_states.get(
            destination_key, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self._image_states.require_owner(source_key, self.queue_family_index)
        self._image_states.require_owner(destination_key, self.queue_family_index)
        if source_state.layout == vk.VK_IMAGE_LAYOUT_UNDEFINED:
            raise VulkanCapabilityError("source image must have a defined layout before copy")

        source_stage = source_state.stage_mask or vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        destination_stage = destination_state.stage_mask or vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        final_destination_layout = (
            vk.VK_IMAGE_LAYOUT_GENERAL
            if destination_host_readable
            else vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )
        final_destination_access = (
            vk.VK_ACCESS_HOST_READ_BIT
            if destination_host_readable
            else vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
        )
        final_destination_stage = (
            vk.VK_PIPELINE_STAGE_HOST_BIT
            if destination_host_readable
            else vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        )

        def record(command_buffer: Any) -> None:
            to_transfer = [
                vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=source_state.access_mask,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                    oldLayout=source_state.layout,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=source.image,
                    subresourceRange=_color_subresource_range(
                        vk, base_array_layer=source_array_layer
                    ),
                ),
                vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=destination_state.access_mask,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=destination_state.layout,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=destination.image,
                    subresourceRange=_color_subresource_range(
                        vk, base_array_layer=destination_array_layer
                    ),
                ),
            ]
            vk.vkCmdPipelineBarrier(
                command_buffer,
                source_stage | destination_stage,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                len(to_transfer),
                to_transfer,
            )
            source_subresource = vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=int(source_array_layer),
                layerCount=1,
            )
            destination_subresource = vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0,
                baseArrayLayer=int(destination_array_layer),
                layerCount=1,
            )
            # vkCmdCopyImage requires identical formats. Use a blit for the
            # validated UNORM <-> sRGB pair as well as for coordinate flips.
            if resize or flip_x or flip_y or not formats_match or source_rect is not None or destination_rect is not None:
                src_x0, src_x1 = (
                    (source_x1, source_x0) if flip_x else (source_x0, source_x1)
                )
                src_y0, src_y1 = (
                    (source_y1, source_y0) if flip_y else (source_y0, source_y1)
                )
                vk.vkCmdBlitImage(
                    command_buffer,
                    source.image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    destination.image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    [
                        vk.VkImageBlit(
                            srcSubresource=source_subresource,
                            srcOffsets=[
                                vk.VkOffset3D(x=src_x0, y=src_y0, z=0),
                                vk.VkOffset3D(x=src_x1, y=src_y1, z=1),
                            ],
                            dstSubresource=destination_subresource,
                            dstOffsets=[
                                vk.VkOffset3D(x=destination_x0, y=destination_y0, z=0),
                                vk.VkOffset3D(x=destination_x1, y=destination_y1, z=1),
                            ],
                        )
                    ],
                    int(filter_mode or vk.VK_FILTER_NEAREST),
                )
            else:
                vk.vkCmdCopyImage(
                    command_buffer,
                    source.image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    destination.image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    [
                        vk.VkImageCopy(
                            srcSubresource=source_subresource,
                            srcOffset=vk.VkOffset3D(x=0, y=0, z=0),
                            dstSubresource=destination_subresource,
                            dstOffset=vk.VkOffset3D(x=0, y=0, z=0),
                            extent=vk.VkExtent3D(
                                width=int(source.width), height=int(source.height), depth=1
                            ),
                        )
                    ],
                )
            to_runtime = [
                vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                    dstAccessMask=source_state.access_mask,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    newLayout=source_state.layout,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=source.image,
                    subresourceRange=_color_subresource_range(
                        vk, base_array_layer=source_array_layer
                    ),
                ),
                vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=final_destination_access,
                oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                newLayout=final_destination_layout,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=destination.image,
                subresourceRange=_color_subresource_range(
                    vk, base_array_layer=destination_array_layer
                ),
                ),
            ]
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                final_destination_stage | destination_stage,
                0,
                0,
                None,
                0,
                None,
                len(to_runtime),
                to_runtime,
            )

        timeline_value = self.submit_on(
            "graphics",
            record,
            wait_for_timeline=wait_for_timeline,
            wait_semaphore=wait_semaphore,
            wait_semaphore_value=wait_semaphore_value,
        )
        self._image_states.update(
            destination_key,
            ImageState(
                layout=final_destination_layout,
                access_mask=final_destination_access,
                stage_mask=final_destination_stage,
                queue_family_index=self.queue_family_index,
            ),
        )
        self._image_states.update(
            source_key,
            source_state,
        )
        return timeline_value

    def copy_image_to_host_buffer(
        self,
        source: Any,
        destination: Any,
        *,
        source_array_layer: int = 0,
        wait_for_timeline: int | None = None,
        wait_semaphore: Any | None = None,
    ) -> int:
        """Copy one tightly packed RGBA image layer into a host-visible buffer."""
        self._ensure_open()
        if getattr(source, "context", self) is not self:
            raise VulkanCapabilityError("Vulkan image belongs to a different context")
        if getattr(destination, "context", self) is not self:
            raise VulkanCapabilityError("Vulkan buffer belongs to a different context")
        if (
            int(source.width) != int(destination.width)
            or int(source.height) != int(destination.height)
        ):
            raise ValueError("image and host readback buffer dimensions must match")
        if int(source_array_layer) < 0:
            raise ValueError("image array layer must not be negative")

        vk = self.vk
        source_key = _cffi_handle_address(vk, source.image)
        source_state = self._image_states.get(
            source_key, undefined_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED
        )
        self._image_states.require_owner(source_key, self.queue_family_index)
        if source_state.layout == vk.VK_IMAGE_LAYOUT_UNDEFINED:
            raise VulkanCapabilityError(
                "source image must have a defined layout before buffer copy"
            )
        source_stage = source_state.stage_mask or vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT

        def record(command_buffer: Any) -> None:
            image_range = _color_subresource_range(
                vk, base_array_layer=source_array_layer
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                source_stage,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=source_state.access_mask,
                        dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                        oldLayout=source_state.layout,
                        newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=source.image,
                        subresourceRange=image_range,
                    )
                ],
            )
            vk.vkCmdCopyImageToBuffer(
                command_buffer,
                source.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                destination.buffer,
                1,
                [
                    vk.VkBufferImageCopy(
                        bufferOffset=0,
                        bufferRowLength=0,
                        bufferImageHeight=0,
                        imageSubresource=vk.VkImageSubresourceLayers(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevel=0,
                            baseArrayLayer=int(source_array_layer),
                            layerCount=1,
                        ),
                        imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        imageExtent=vk.VkExtent3D(
                            width=int(source.width),
                            height=int(source.height),
                            depth=1,
                        ),
                    )
                ],
            )
            vk.vkCmdPipelineBarrier(
                command_buffer,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_HOST_BIT | source_stage,
                0,
                0,
                None,
                1,
                [
                    vk.VkBufferMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        buffer=destination.buffer,
                        offset=0,
                        size=destination.size,
                    )
                ],
                1,
                [
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                        dstAccessMask=source_state.access_mask,
                        oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        newLayout=source_state.layout,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=source.image,
                        subresourceRange=image_range,
                    )
                ],
            )

        return self.submit_on(
            "graphics",
            record,
            wait_for_timeline=wait_for_timeline,
            wait_semaphore=wait_semaphore,
        )

    def submit(self, record: Callable[[Any], None]) -> None:
        self.submit_on("graphics", record)

    def submit_on(
        self,
        role: str,
        record: Callable[[Any], None],
        *,
        wait_for_timeline: int | None = None,
        wait_semaphore: Any | None = None,
        wait_semaphore_value: int | None = None,
        signal_semaphore: Any | None = None,
        signal_semaphore_value: int | None = None,
        on_submit_profile: Callable[[dict[str, float]], None] | None = None,
    ) -> int:
        try:
            return self._submit_on_unchecked(
                role,
                record,
                wait_for_timeline=wait_for_timeline,
                wait_semaphore=wait_semaphore,
                wait_semaphore_value=wait_semaphore_value,
                signal_semaphore=signal_semaphore,
                signal_semaphore_value=signal_semaphore_value,
                on_submit_profile=on_submit_profile,
            )
        except Exception as exc:
            # Device loss may surface from any command-buffer or queue call,
            # not only the fence wait observed in the original crash.
            self.mark_device_lost(exc)
            raise

    def _submit_on_unchecked(
        self,
        role: str,
        record: Callable[[Any], None],
        *,
        wait_for_timeline: int | None = None,
        wait_semaphore: Any | None = None,
        wait_semaphore_value: int | None = None,
        signal_semaphore: Any | None = None,
        signal_semaphore_value: int | None = None,
        on_submit_profile: Callable[[dict[str, float]], None] | None = None,
    ) -> int:
        with self._lock:
            self._ensure_open()
            vk = self.vk
            queue_role = str(role).lower()
            if queue_role not in ("graphics", "compute", "transfer"):
                raise ValueError(f"unknown Vulkan queue role: {role}")
            frame_index = self._frame_indices[queue_role]
            frame = self._frame_contexts[frame_index]
            queue_resources = frame.queue_resources[queue_role]
            # Reuse is bounded by the frame fence instead of allocating per submit.
            try:
                fence_wait_started = perf_counter()
                wait_result = vk.vkWaitForFences(
                    self.device, 1, [queue_resources.fence], vk.VK_TRUE, 10_000_000_000
                )
                fence_wait_seconds = perf_counter() - fence_wait_started
            except Exception as exc:
                self.mark_device_lost(exc)
                raise
            timeout = getattr(vk, "VK_TIMEOUT", None)
            if timeout is not None and wait_result == timeout:
                raise VulkanCapabilityError(
                    f"timed out waiting for {queue_role} FrameContext fence"
                )
            vk.vkResetFences(self.device, 1, [queue_resources.fence])
            vk.vkResetCommandBuffer(queue_resources.command_buffer, 0)
            begin_info = vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            )
            vk.vkBeginCommandBuffer(queue_resources.command_buffer, begin_info)
            record_started = perf_counter()
            try:
                record(queue_resources.command_buffer)
                vk.vkEndCommandBuffer(queue_resources.command_buffer)
            except Exception:
                try:
                    vk.vkEndCommandBuffer(queue_resources.command_buffer)
                except Exception:
                    pass
                raise
            record_seconds = perf_counter() - record_started
            self._timeline_value += 1
            queue_resources.timeline_value = self._timeline_value
            try:
                queue_submit_started = perf_counter()
                self._submit_frame(
                    queue=self.get_queue(role),
                    command_buffer=queue_resources.command_buffer,
                    fence=queue_resources.fence,
                    timeline_value=queue_resources.timeline_value,
                    wait_timeline_value=wait_for_timeline,
                    wait_semaphore=wait_semaphore,
                    wait_semaphore_value=wait_semaphore_value,
                    signal_semaphore=signal_semaphore,
                    signal_semaphore_value=signal_semaphore_value,
                )
                queue_submit_seconds = perf_counter() - queue_submit_started
            except Exception as exc:
                self.mark_device_lost(exc)
                raise
            self._frame_indices[queue_role] = (
                frame_index + 1
            ) % self.frame_context_count
            if on_submit_profile is not None:
                try:
                    on_submit_profile(
                        {
                            "fence_wait": max(0.0, fence_wait_seconds),
                            "record": max(0.0, record_seconds),
                            "queue_submit": max(0.0, queue_submit_seconds),
                        }
                    )
                except Exception:
                    pass
            return queue_resources.timeline_value

    def wait_idle(self) -> None:
        with self._lock:
            if not self._closed and self.device is not None and not self._device_lost:
                try:
                    self.vk.vkDeviceWaitIdle(self.device)
                except Exception as exc:
                    self.mark_device_lost(exc)
                    raise

    def wait_for_timeline(self, value: int, *, timeout_ns: int = 10_000_000_000) -> None:
        """Wait on this context's timeline without forcing a device-wide idle."""
        target = int(value)
        if target <= 0:
            return
        with self._lock:
            self._ensure_open()
            if self._timeline_semaphore is None:
                self.vk.vkDeviceWaitIdle(self.device)
                return
            wait_fn = getattr(self.vk, "vkWaitSemaphores", None)
            wait_info_type = getattr(self.vk, "VkSemaphoreWaitInfo", None)
            wait_info_structure = getattr(self.vk, "VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO", None)
            if wait_fn is None or wait_info_type is None or wait_info_structure is None:
                self.vk.vkDeviceWaitIdle(self.device)
                return
            try:
                result = wait_fn(
                    self.device,
                    wait_info_type(
                        sType=wait_info_structure,
                        semaphoreCount=1,
                        pSemaphores=[self._timeline_semaphore],
                        pValues=[target],
                    ),
                    int(timeout_ns),
                )
            except Exception as exc:
                self.mark_device_lost(exc)
                raise
            if result is not None and int(result) != int(self.vk.VK_SUCCESS):
                raise VulkanCapabilityError(
                    f"timed out waiting for Vulkan timeline value {target}: {result}"
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            vk = self.vk
            try:
                if self.device is not None and not self._device_lost:
                    try:
                        vk.vkDeviceWaitIdle(self.device)
                    except Exception as exc:
                        self.mark_device_lost(exc)
                        if not self._device_lost:
                            raise
                registry = getattr(self, "_external_image_registry", None)
                if registry is not None:
                    if self._device_lost:
                        registry.discard()
                    else:
                        try:
                            registry.close()
                        except Exception:
                            registry.discard()
            finally:
                if self.device is not None:
                    if self._timeline_semaphore is not None:
                        vk.vkDestroySemaphore(self.device, self._timeline_semaphore, None)
                    for frame in self._frame_contexts:
                        for resources in frame.queue_resources.values():
                            vk.vkDestroyFence(self.device, resources.fence, None)
                            vk.vkDestroyCommandPool(self.device, resources.command_pool, None)
                    if self._owns_device:
                        vk.vkDestroyDevice(self.device, None)
                if self.instance is not None and self._owns_instance:
                    vk.vkDestroyInstance(self.instance, None)
                self._image_states.clear()
                self._external_image_registry = None
                self._command_buffer = None
                self._command_pool = None
                self._fence = None
                self._timeline_semaphore = None
                self._frame_contexts.clear()
                self.queue = None
                self.device = None
                self.physical_device = None
                self.instance = None
                self._closed = True

    def __enter__(self) -> "VulkanContext":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _create_command_resources(self) -> None:
        vk = self.vk
        fence_info = vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
        )
        for _ in range(self.frame_context_count):
            resources: dict[str, VulkanQueueFrameResources] = {}
            for role in ("graphics", "compute", "transfer"):
                pool_info = vk.VkCommandPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    queueFamilyIndex=self.queue_family(role),
                )
                command_pool = vk.vkCreateCommandPool(self.device, pool_info, None)
                allocation_info = vk.VkCommandBufferAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    commandPool=command_pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                )
                command_buffer = vk.vkAllocateCommandBuffers(self.device, allocation_info)[0]
                resources[role] = VulkanQueueFrameResources(
                    command_pool,
                    command_buffer,
                    vk.vkCreateFence(self.device, fence_info, None),
                )
            graphics = resources["graphics"]
            self._frame_contexts.append(
                VulkanFrameContext(
                    graphics.command_pool,
                    graphics.command_buffer,
                    graphics.fence,
                    queue_resources=resources,
                )
            )
        self._command_pool = self._frame_contexts[0].command_pool
        self._command_buffer = self._frame_contexts[0].command_buffer
        self._fence = self._frame_contexts[0].fence

    def _submit_frame(
        self,
        *,
        queue: Any,
        command_buffer: Any,
        fence: Any,
        timeline_value: int,
        wait_timeline_value: int | None = None,
        wait_semaphore: Any | None = None,
        wait_semaphore_value: int | None = None,
        signal_semaphore: Any | None = None,
        signal_semaphore_value: int | None = None,
    ) -> None:
        vk = self.vk
        wait_binary = (
            list(wait_semaphore)
            if isinstance(wait_semaphore, (tuple, list))
            else ([] if wait_semaphore is None else [wait_semaphore])
        )
        signal_binary = (
            list(signal_semaphore)
            if isinstance(signal_semaphore, (tuple, list))
            else ([] if signal_semaphore is None else [signal_semaphore])
        )
        if wait_semaphore_value is not None and len(wait_binary) != 1:
            raise VulkanCapabilityError(
                "timeline semaphore value requires exactly one wait semaphore"
            )
        if signal_semaphore_value is not None and len(signal_binary) != 1:
            raise VulkanCapabilityError(
                "timeline semaphore value requires exactly one signal semaphore"
            )
        wait_external_values = [0] * len(wait_binary)
        signal_external_values = [0] * len(signal_binary)
        if wait_semaphore_value is not None:
            wait_external_values[0] = int(wait_semaphore_value)
        if signal_semaphore_value is not None:
            signal_external_values[0] = int(signal_semaphore_value)
        if wait_timeline_value is not None and self._timeline_semaphore is None:
            raise VulkanCapabilityError(
                "timeline wait requested but the Vulkan context has no timeline semaphore"
            )
        if self._timeline_semaphore is not None and _supports_submit2(
            vk,
            self.device_info.api_version,
            self.device_info.synchronization2_enabled,
        ):
            command_info = vk.VkCommandBufferSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                commandBuffer=command_buffer,
                deviceMask=1,
            )
            signal_info = vk.VkSemaphoreSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                semaphore=self._timeline_semaphore,
                value=timeline_value,
                stageMask=_pipeline_stage_2_all_commands(vk),
                deviceIndex=0,
            )
            wait_infos = []
            for semaphore, semaphore_value in zip(
                wait_binary, wait_external_values, strict=True
            ):
                wait_infos.append(
                    vk.VkSemaphoreSubmitInfo(
                        sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                        semaphore=semaphore,
                        value=semaphore_value,
                        stageMask=_pipeline_stage_2_all_commands(vk),
                        deviceIndex=0,
                    )
                )
            if wait_timeline_value is not None:
                wait_infos.append(
                    vk.VkSemaphoreSubmitInfo(
                        sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                        semaphore=self._timeline_semaphore,
                        value=int(wait_timeline_value),
                        stageMask=_pipeline_stage_2_all_commands(vk),
                        deviceIndex=0,
                    )
                )
            signal_infos = [signal_info]
            for semaphore, semaphore_value in zip(
                signal_binary, signal_external_values, strict=True
            ):
                signal_infos.append(
                    vk.VkSemaphoreSubmitInfo(
                        sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                        semaphore=semaphore,
                        value=semaphore_value,
                        stageMask=_pipeline_stage_2_all_commands(vk),
                        deviceIndex=0,
                    )
                )
            submit_info = vk.VkSubmitInfo2(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                waitSemaphoreInfoCount=len(wait_infos),
                pWaitSemaphoreInfos=wait_infos or None,
                commandBufferInfoCount=1,
                pCommandBufferInfos=[command_info],
                signalSemaphoreInfoCount=len(signal_infos),
                pSignalSemaphoreInfos=signal_infos,
            )
            vk.vkQueueSubmit2(queue, 1, [submit_info], fence)
            return

        wait_semaphores = list(wait_binary)
        wait_stage_masks = [
            vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT for _semaphore in wait_binary
        ]
        wait_values = list(wait_external_values)
        if wait_timeline_value is not None:
            wait_semaphores.append(self._timeline_semaphore)
            wait_stage_masks.append(vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
            wait_values.append(int(wait_timeline_value))
        signal_semaphores = []
        signal_values = []
        if self._timeline_semaphore is not None:
            signal_semaphores.append(self._timeline_semaphore)
            signal_values.append(timeline_value)
        signal_semaphores.extend(signal_binary)
        signal_values.extend(signal_external_values)
        timeline_info = None
        if any(wait_values) or any(signal_values):
            timeline_info = vk.VkTimelineSemaphoreSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                waitSemaphoreValueCount=len(wait_values),
                pWaitSemaphoreValues=wait_values or None,
                signalSemaphoreValueCount=len(signal_values),
                pSignalSemaphoreValues=signal_values,
            )
        submit_info = vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext=timeline_info,
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
            waitSemaphoreCount=len(wait_semaphores),
            pWaitSemaphores=wait_semaphores or None,
            pWaitDstStageMask=wait_stage_masks or None,
            signalSemaphoreCount=len(signal_semaphores),
            pSignalSemaphores=signal_semaphores or None,
        )
        vk.vkQueueSubmit(queue, 1, [submit_info], fence)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VulkanContext is closed")
        if self._device_lost:
            raise VulkanCapabilityError("Vulkan device is lost")

    @property
    def device_lost(self) -> bool:
        return bool(self._device_lost)

    @property
    def device_lost_error(self) -> str | None:
        return self._device_lost_error

    def mark_device_lost(self, exc: BaseException) -> bool:
        """Latch device loss and report whether *exc* represented that state."""
        if not is_vulkan_device_lost_error(exc):
            return False
        self._device_lost = True
        if self._device_lost_error is None:
            self._device_lost_error = str(exc) or type(exc).__name__
        return True


def _import_vulkan() -> Any:
    try:
        import vulkan as vk
    except (ImportError, OSError) as exc:
        raise VulkanUnavailableError(
            "Python Vulkan bindings or the Vulkan loader are unavailable"
        ) from exc
    return vk


def _create_timeline_semaphore(vk: Any, device: Any) -> Any:
    semaphore_type = vk.VkSemaphoreTypeCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        semaphoreType=vk.VK_SEMAPHORE_TYPE_TIMELINE,
        initialValue=0,
    )
    create_info = vk.VkSemaphoreCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        pNext=semaphore_type,
    )
    return vk.vkCreateSemaphore(device, create_info, None)


def _supports_submit2(
    vk: Any, api_version: int, synchronization2_enabled: bool
) -> bool:
    if int(api_version) < make_vulkan_version(1, 3, 0) or not synchronization2_enabled:
        return False
    return all(
        hasattr(vk, name)
        for name in (
            "vkQueueSubmit2",
            "VkCommandBufferSubmitInfo",
            "VkSemaphoreSubmitInfo",
            "VkSubmitInfo2",
            "VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO",
            "VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO",
            "VK_STRUCTURE_TYPE_SUBMIT_INFO_2",
        )
    )


def _pipeline_stage_2_all_commands(vk: Any) -> int:
    value = getattr(vk, "VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT", None)
    if value is None:
        value = getattr(vk, "VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR")
    return int(value)


def _enumerate_names(properties: Iterable[Any], field: str) -> set[str]:
    names: set[str] = set()
    for item in properties:
        value = getattr(item, field)
        if isinstance(value, bytes):
            value = value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        names.add(str(value))
    return names


def _require_names(
    label: str,
    required: Iterable[str],
    available: set[str],
) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(str(name) for name in required))
    missing = [name for name in requested if name not in available]
    if missing:
        raise VulkanCapabilityError(f"Missing {label}s: {', '.join(missing)}")
    return requested


def _select_physical_device(vk: Any, instance: Any) -> tuple[Any, QueueFamilySelection]:
    devices = vk.vkEnumeratePhysicalDevices(instance)
    if not devices:
        raise VulkanCapabilityError("No Vulkan physical device is available")
    candidates: list[tuple[int, Any, QueueFamilySelection]] = []
    for physical_device in devices:
        queue_families = _find_queue_families(vk, physical_device)
        if queue_families is None:
            continue
        properties = vk.vkGetPhysicalDeviceProperties(physical_device)
        score = {
            vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: 300,
            vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: 200,
            vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: 100,
            vk.VK_PHYSICAL_DEVICE_TYPE_CPU: 10,
        }.get(int(properties.deviceType), 0)
        candidates.append((score, physical_device, queue_families))
    if not candidates:
        raise VulkanCapabilityError("No Vulkan device exposes a graphics queue")
    _, physical_device, queue_families = max(candidates, key=lambda item: item[0])
    return physical_device, queue_families


def find_graphics_queue_family(vk: Any, physical_device: Any) -> int:
    queue_family_index = _find_graphics_queue_family(vk, physical_device)
    if queue_family_index is None:
        raise VulkanCapabilityError("OpenXR Vulkan device has no graphics queue")
    return queue_family_index


def _find_graphics_queue_family(vk: Any, physical_device: Any) -> int | None:
    for index, properties in enumerate(
        vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device)
    ):
        if properties.queueCount and properties.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
            return index
    return None


def _find_queue_families(vk: Any, physical_device: Any) -> QueueFamilySelection | None:
    families = list(vk.vkGetPhysicalDeviceQueueFamilyProperties(physical_device))
    graphics = next(
        (
            index
            for index, properties in enumerate(families)
            if properties.queueCount and properties.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
        ),
        None,
    )
    if graphics is None:
        return None

    compute = next(
        (
            index
            for index, properties in enumerate(families)
            if properties.queueCount
            and properties.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
            and not properties.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
        ),
        graphics,
    )
    transfer_bit = getattr(vk, "VK_QUEUE_TRANSFER_BIT", 0)
    transfer = next(
        (
            index
            for index, properties in enumerate(families)
            if properties.queueCount
            and transfer_bit
            and properties.queueFlags & transfer_bit
            and not properties.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
            and not properties.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
        ),
        compute,
    )
    return QueueFamilySelection(graphics, compute, transfer)


def _device_info(
    vk: Any,
    physical_device: Any,
    queue_families: QueueFamilySelection,
    *,
    timeline_semaphore_enabled: bool = False,
    synchronization2_enabled: bool = False,
) -> VulkanDeviceInfo:
    properties = vk.vkGetPhysicalDeviceProperties(physical_device)
    adapter_luid = _query_adapter_luid(vk, physical_device)
    return VulkanDeviceInfo(
        name=str(properties.deviceName),
        api_version=int(properties.apiVersion),
        driver_version=int(properties.driverVersion),
        vendor_id=int(properties.vendorID),
        device_id=int(properties.deviceID),
        device_type=int(properties.deviceType),
        queue_family_index=int(queue_families.graphics),
        compute_queue_family_index=int(queue_families.compute),
        transfer_queue_family_index=int(queue_families.transfer),
        timeline_semaphore_enabled=bool(timeline_semaphore_enabled),
        synchronization2_enabled=bool(synchronization2_enabled),
        adapter_luid=adapter_luid,
    )


def _query_adapter_luid(vk: Any, physical_device: Any) -> int:
    """Read VkPhysicalDeviceIDProperties.deviceLUID when the binding exposes it."""
    properties2_type = getattr(vk, "VkPhysicalDeviceProperties2", None)
    identity_type = getattr(vk, "VkPhysicalDeviceIDProperties", None)
    get_properties2 = getattr(vk, "vkGetPhysicalDeviceProperties2", None)
    if properties2_type is None or identity_type is None or get_properties2 is None:
        return 0
    try:
        identity = identity_type(
            sType=getattr(vk, "VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES"),
        )
        properties2 = properties2_type(
            sType=getattr(vk, "VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2"),
            pNext=identity,
        )
        get_properties2(physical_device, properties2)
        if not bool(getattr(identity, "deviceLUIDValid", False)):
            return 0
        return int.from_bytes(bytes(vk.ffi.buffer(identity.deviceLUID, 8)), "little")
    except Exception:
        return 0


def _loader_api_version(vk: Any) -> int:
    enumerate_version = getattr(vk, "vkEnumerateInstanceVersion", None)
    if enumerate_version is None:
        return make_vulkan_version(1, 0, 0)
    return int(enumerate_version())


def _require_timeline_semaphore_features(
    vk: Any, physical_device: Any, *, require_multiview: bool = False
) -> tuple[Any, bool]:
    feature_type = getattr(vk, "VkPhysicalDeviceTimelineSemaphoreFeatures", None)
    features2_type = getattr(vk, "VkPhysicalDeviceFeatures2", None)
    get_features2 = getattr(vk, "vkGetPhysicalDeviceFeatures2", None)
    if feature_type is None or features2_type is None or get_features2 is None:
        raise VulkanCapabilityError(
            "Vulkan binding does not expose physical-device feature2 queries"
        )

    multiview_type = getattr(vk, "VkPhysicalDeviceMultiviewFeatures", None)
    if require_multiview and multiview_type is None:
        raise VulkanCapabilityError("Vulkan binding does not expose multiview features")
    multiview_supported = multiview_type() if require_multiview else None
    synchronization_type = getattr(vk, "VkPhysicalDeviceSynchronization2Features", None)
    synchronization_supported = (
        synchronization_type(pNext=multiview_supported)
        if synchronization_type
        else multiview_supported
    )
    supported = feature_type(pNext=synchronization_supported)
    features2 = features2_type(pNext=supported)
    get_features2(physical_device, features2)
    if not bool(supported.timelineSemaphore):
        raise VulkanCapabilityError(
            "Vulkan device does not support timelineSemaphore"
        )
    if require_multiview and not bool(multiview_supported.multiview):
        raise VulkanCapabilityError("Vulkan device does not support multiview")

    synchronization2_enabled = bool(
        synchronization_supported is not None
        and getattr(synchronization_supported, "synchronization2", vk.VK_FALSE)
    )
    enabled_timeline = feature_type(timelineSemaphore=vk.VK_TRUE)
    enabled_features = enabled_timeline
    if require_multiview:
        enabled_features = multiview_type(
            multiview=vk.VK_TRUE,
            pNext=enabled_features,
        )
    if synchronization2_enabled:
        enabled_sync = synchronization_type(
            synchronization2=vk.VK_TRUE,
            pNext=enabled_features,
        )
        return enabled_sync, True
    return enabled_features, False


def _create_device(
    vk: Any,
    physical_device: Any,
    queue_families: QueueFamilySelection,
    required_extensions: Iterable[str],
) -> tuple[Any, bool]:
    properties = vk.vkGetPhysicalDeviceProperties(physical_device)
    if int(properties.apiVersion) < MIN_VULKAN_API_VERSION:
        raise VulkanCapabilityError(
            "Vulkan device API version is below the required Vulkan 1.2"
        )

    extension_names = tuple(required_extensions)
    queue_infos = [
        vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=int(queue_family_index),
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        for queue_family_index in dict.fromkeys(
            (queue_families.graphics, queue_families.compute, queue_families.transfer)
        )
    ]
    timeline_features, synchronization2_enabled = _require_timeline_semaphore_features(
        vk, physical_device
    )
    device_info = vk.VkDeviceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        pNext=timeline_features,
        queueCreateInfoCount=len(queue_infos),
        pQueueCreateInfos=queue_infos,
        enabledExtensionCount=len(extension_names),
        ppEnabledExtensionNames=list(extension_names) or None,
    )
    return (
        vk.vkCreateDevice(physical_device, device_info, None),
        synchronization2_enabled,
    )


def _color_subresource_range(
    vk: Any, *, base_array_layer: int = 0, layer_count: int = 1
) -> Any:
    return vk.VkImageSubresourceRange(
        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
        baseMipLevel=0,
        levelCount=1,
        baseArrayLayer=int(base_array_layer),
        layerCount=int(layer_count),
    )


def _depth_subresource_range(vk: Any, *, aspect_mask: int) -> Any:
    return vk.VkImageSubresourceRange(
        aspectMask=int(aspect_mask),
        baseMipLevel=0,
        levelCount=1,
        baseArrayLayer=0,
        layerCount=1,
    )


def _cffi_handle_address(vk: Any, handle: Any) -> int:
    return int(vk.ffi.cast("uintptr_t", handle))
