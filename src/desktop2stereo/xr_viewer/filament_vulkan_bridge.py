from __future__ import annotations

import ctypes
import json
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Iterable


class FilamentBridgeError(RuntimeError):
    pass


_GLB_JSON_CHUNK = 0x4E4F534A
_DYNAMIC_LIGHT_MATERIAL_PREFIX = "D2S_REFLECTIVE__"
_PRESERVE_UNLIT_NAME_PARTS = (
    "skybox", "fakelight", "screen", "logo", "fire", "emmi",
    "emissive", "neon", "warning", "glass", "clock", "maxanimate",
)
_UNLIT_ONLY_MATERIAL_EXTENSIONS = (
    "KHR_materials_clearcoat",
    "KHR_materials_dispersion",
    "KHR_materials_ior",
    "KHR_materials_pbrSpecularGlossiness",
    "KHR_materials_sheen",
    "KHR_materials_specular",
    "KHR_materials_transmission",
    "KHR_materials_volume",
)


def _prepare_environment_glb_for_dynamic_lighting(
    data: bytes,
) -> tuple[bytes, tuple[str, ...]]:
    """Make baked room surfaces lit while retaining authored emissive assets."""
    payload = bytes(data)
    if len(payload) < 20 or payload[:4] != b"glTF":
        return payload, ()
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if version != 2 or declared_length != len(payload):
        return payload, ()

    chunks: list[tuple[int, bytes]] = []
    document = None
    json_chunk_index = None
    offset = 12
    while offset + 8 <= len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_length
        if chunk_end > len(payload):
            return payload, ()
        chunk = payload[chunk_start:chunk_end]
        if chunk_type == _GLB_JSON_CHUNK and document is None:
            try:
                document = json.loads(chunk.decode("utf-8").rstrip("\x00 "))
            except (UnicodeDecodeError, ValueError):
                return payload, ()
            json_chunk_index = len(chunks)
        chunks.append((chunk_type, chunk))
        offset = chunk_end
    if offset != len(payload) or document is None or json_chunk_index is None:
        return payload, ()

    converted = []
    for index, material in enumerate(document.get("materials", ())):
        if not isinstance(material, dict):
            continue
        extensions = material.get("extensions")
        if not isinstance(extensions, dict) or "KHR_materials_unlit" not in extensions:
            continue
        name = str(material.get("name", f"material_{index}"))
        lowered = name.casefold()
        pbr = material.get("pbrMetallicRoughness")
        has_base_texture = isinstance(pbr, dict) and isinstance(
            pbr.get("baseColorTexture"), dict
        )
        preserve = (
            not has_base_texture
            or any(part in lowered for part in _PRESERVE_UNLIT_NAME_PARTS)
            or lowered.startswith("line_")
            or "_light_" in lowered
        )
        if preserve:
            continue
        extensions.pop("KHR_materials_unlit", None)
        # These inputs were ignored by the authored unlit material. Normalize
        # the converted surface to the small metallic/roughness contract that
        # the native custom provider exposes to glTFio.
        for extension_name in _UNLIT_ONLY_MATERIAL_EXTENSIONS:
            extensions.pop(extension_name, None)
        if not extensions:
            material.pop("extensions", None)
        pbr.pop("metallicRoughnessTexture", None)
        material.pop("normalTexture", None)
        material.pop("occlusionTexture", None)
        pbr.setdefault("metallicFactor", 0.0)
        pbr.setdefault("roughnessFactor", 0.85)
        # These rooms were authored as baked unlit scenes. Removing the unlit
        # extension without retaining that baked contribution makes the room
        # nearly black beside its still-unlit skybox. Reuse the authored base
        # texture as the emissive baseline, then let Filament add ambient and
        # segmented screen lights through the newly enabled PBR response.
        material.setdefault("emissiveTexture", dict(pbr["baseColorTexture"]))
        material.setdefault("emissiveFactor", [1.0, 1.0, 1.0])
        # The native MaterialProvider receives the glTF material label before
        # ResourceLoader binds its textures. Prefix only the converted baked
        # room surfaces so it can select the Lit custom-surface material while
        # preserving glTFio's normal texture upload and binding path.
        material["name"] = _DYNAMIC_LIGHT_MATERIAL_PREFIX + name
        converted.append(name)

    if not converted:
        return payload, ()
    still_uses_unlit = any(
        isinstance(material, dict)
        and "KHR_materials_unlit" in material.get("extensions", {})
        for material in document.get("materials", ())
    )
    if not still_uses_unlit:
        for key in ("extensionsUsed", "extensionsRequired"):
            values = document.get(key)
            if isinstance(values, list):
                document[key] = [
                    value for value in values if value != "KHR_materials_unlit"
                ]
                if not document[key]:
                    document.pop(key, None)

    json_data = json.dumps(
        document, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    json_data += b" " * ((-len(json_data)) % 4)
    chunks[json_chunk_index] = (_GLB_JSON_CHUNK, json_data)
    rebuilt = bytearray(struct.pack("<4sII", magic, version, 0))
    for chunk_type, chunk in chunks:
        rebuilt += struct.pack("<II", len(chunk), chunk_type)
        rebuilt += chunk
    struct.pack_into("<I", rebuilt, 8, len(rebuilt))
    return bytes(rebuilt), tuple(converted)


class _VulkanCreateInfo(ctypes.Structure):
    _fields_ = [
        ("instance", ctypes.c_void_p),
        ("physical_device", ctypes.c_void_p),
        ("device", ctypes.c_void_p),
        ("graphics_queue_family_index", ctypes.c_uint32),
        ("graphics_queue_index", ctypes.c_uint32),
    ]


class _FilamentLightingConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("environment_ambient_color", ctypes.c_float * 3),
        ("environment_ambient_intensity_lux", ctypes.c_float),
        ("controller_ambient_color", ctypes.c_float * 3),
        ("controller_ambient_intensity_lux", ctypes.c_float),
        ("controller_ambient_enabled", ctypes.c_int32),
        ("head_light_color", ctypes.c_float * 3),
        ("head_light_intensity_candela", ctypes.c_float),
        ("head_light_offset", ctypes.c_float * 3),
        ("head_light_falloff", ctypes.c_float),
        ("head_light_cast_shadows", ctypes.c_int32),
        ("top_light_color", ctypes.c_float * 3),
        ("top_light_intensity_candela", ctypes.c_float),
        ("top_light_offset", ctypes.c_float * 3),
        ("top_light_falloff", ctypes.c_float),
        ("top_light_cast_shadows", ctypes.c_int32),
    ]


def default_bridge_path() -> Path:
    platforms = {
        "win32": ("windows", "filament_bridge.dll"),
        "darwin": ("macos", "libfilament_bridge.dylib"),
        "linux": ("linux", "libfilament_bridge.so"),
    }
    try:
        platform_dir, name = platforms[sys.platform]
    except KeyError as exc:
        raise FilamentBridgeError(
            f"unsupported platform for Filament Bridge: {sys.platform}"
        ) from exc
    return Path(__file__).resolve().parent / "native" / platform_dir / name


class FilamentVulkanBridge:
    """ctypes wrapper for the Python-owned OpenXR Vulkan session handles."""

    def __init__(self, library_path: str | Path | None = None) -> None:
        path = Path(library_path) if library_path else default_bridge_path()
        try:
            self._library = ctypes.CDLL(str(path))
        except OSError as exc:
            raise FilamentBridgeError(f"unable to load Filament Bridge: {path}") from exc
        self._controller_abi_available = False
        self._controller_visibility_abi_available = False
        self._controller_material_override_abi_available = False
        self._laser_abi_available = False
        self._controller_guide_abi_available = False
        self._text_overlay_abi_available = False
        self._vulkan_external_image_abi_available = False
        self._depth_output_abi_available = False
        self._depth_swapchain_abi_available = False
        self._passthrough_backdrop_abi_available = False
        self._ambient_light_abi_available = False
        self._controller_ambient_light_abi_available = False
        self._lighting_config_abi_available = False
        self._controller_screen_light_abi_available = False
        self._environment_screen_lights_abi_available = False
        self._environment_screen_area_light_abi_available = False
        self._finished_drawing_semaphore_abi_available = False
        self._image_ready_semaphore_abi_available = False
        self._controller_overlay_abi_available = False
        self._controller_overlay_batch_abi_available = False
        self._controller_composition_layer_abi_available = False
        self._background_frame_abi_available = False
        self._async_submit_abi_available = False
        self._stereo_batch_submit_abi_available = False
        self._environment_unload_abi_available = False
        self._multiview_abi_version = 0
        self._multiview_depth_swapchain_abi_available = False
        self._configure_abi()
        self._handle: ctypes.c_void_p | None = None
        self._owner_thread_id: int | None = None

    @property
    def handle(self) -> int:
        return int(self._handle.value or 0) if self._handle is not None else 0

    @property
    def controller_abi_available(self) -> bool:
        return self._controller_abi_available

    @property
    def laser_abi_available(self) -> bool:
        return self._laser_abi_available

    @property
    def controller_guide_abi_available(self) -> bool:
        return self._controller_guide_abi_available

    @property
    def text_overlay_abi_available(self) -> bool:
        return self._text_overlay_abi_available

    @property
    def controller_visibility_abi_available(self) -> bool:
        return self._controller_visibility_abi_available

    @property
    def controller_material_override_abi_available(self) -> bool:
        return self._controller_material_override_abi_available

    @property
    def environment_screen_lights_abi_available(self) -> bool:
        return self._environment_screen_lights_abi_available

    @property
    def environment_screen_area_light_abi_available(self) -> bool:
        return self._environment_screen_area_light_abi_available

    @property
    def vulkan_external_image_abi_available(self) -> bool:
        return self._vulkan_external_image_abi_available

    @property
    def depth_output_abi_available(self) -> bool:
        return self._depth_output_abi_available

    @property
    def depth_swapchain_abi_available(self) -> bool:
        return self._depth_swapchain_abi_available

    @property
    def passthrough_backdrop_abi_available(self) -> bool:
        return self._passthrough_backdrop_abi_available

    @property
    def async_submit_abi_available(self) -> bool:
        return self._async_submit_abi_available

    @property
    def stereo_batch_submit_abi_available(self) -> bool:
        return self._stereo_batch_submit_abi_available

    @property
    def controller_overlay_abi_available(self) -> bool:
        return self._controller_overlay_abi_available

    @property
    def controller_overlay_batch_abi_available(self) -> bool:
        return self._controller_overlay_batch_abi_available

    @property
    def controller_composition_layer_abi_available(self) -> bool:
        return self._controller_composition_layer_abi_available

    @property
    def background_frame_abi_available(self) -> bool:
        return self._background_frame_abi_available

    @property
    def multiview_abi_available(self) -> bool:
        return self._multiview_abi_version >= 2

    @property
    def multiview_depth_swapchain_abi_available(self) -> bool:
        return self._multiview_depth_swapchain_abi_available

    @property
    def multiview_supported(self) -> bool:
        if not self.multiview_abi_available or not self.loaded:
            return False
        return bool(
            self._library.filament_bridge_multiview_supported(self._handle)
        )

    @property
    def loaded(self) -> bool:
        return self._handle is not None and bool(self._handle.value)

    def create(
        self,
        *,
        instance: Any,
        physical_device: Any,
        device: Any,
        queue_family_index: int,
        queue_index: int = 0,
    ) -> None:
        if self.loaded:
            self._ensure_owner_thread()
            return
        self._owner_thread_id = threading.get_ident()
        info = _VulkanCreateInfo(
            instance=_as_pointer_value(instance),
            physical_device=_as_pointer_value(physical_device),
            device=_as_pointer_value(device),
            graphics_queue_family_index=int(queue_family_index),
            graphics_queue_index=int(queue_index),
        )
        handle = self._library.filament_bridge_create_vulkan(ctypes.byref(info))
        if not handle:
            raise FilamentBridgeError("Filament Bridge returned a null handle")
        self._handle = ctypes.c_void_p(handle)
        self._raise_if_error()

    def create_swapchain(
        self,
        image_handles: Iterable[Any],
        *,
        format: int,
        width: int,
        height: int,
    ) -> None:
        self._ensure_loaded()
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if not values:
            raise ValueError("Filament swapchain requires at least one VkImage")
        array_type = ctypes.c_void_p * len(values)
        result = self._library.filament_bridge_create_swapchain(
            self._handle,
            array_type(*values),
            len(values),
            int(format),
            int(width),
            int(height),
        )
        self._check_result(result, "create_swapchain")

    def create_eye_swapchain(
        self,
        eye_index: int,
        image_handles: Iterable[Any],
        *,
        format: int,
        width: int,
        height: int,
    ) -> None:
        self._ensure_loaded()
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if int(eye_index) not in (0, 1) or not values:
            raise ValueError("eye_index must be 0 or 1 and swapchain must not be empty")
        array_type = ctypes.c_void_p * len(values)
        self._check_result(
            self._library.filament_bridge_create_eye_swapchain(
                self._handle,
                int(eye_index),
                array_type(*values),
                len(values),
                int(format),
                int(width),
                int(height),
            ),
            "create_eye_swapchain",
        )

    def create_eye_swapchain_with_depth(
        self,
        eye_index: int,
        image_handles: Iterable[Any],
        *,
        format: int,
        width: int,
        height: int,
        depth_image: Any,
        depth_format: int,
    ) -> None:
        self._ensure_loaded()
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if int(eye_index) not in (0, 1) or not values:
            raise ValueError("eye_index must be 0 or 1 and swapchain must not be empty")
        if depth_image is None:
            raise ValueError("depth_image must not be empty")
        function = getattr(
            self._library, "filament_bridge_create_eye_swapchain_with_depth", None
        )

    def get_depth_attachment(self, eye_index: int) -> tuple[int, int] | None:
        """Return the borrowed native depth image and format when ready."""
        self._ensure_loaded()
        function = getattr(self._library, "filament_bridge_get_depth_attachment", None)
        if function is None:
            return None
        image = ctypes.c_void_p()
        format_value = ctypes.c_int32()
        if not function(
            self._handle,
            int(eye_index),
            ctypes.byref(image),
            ctypes.byref(format_value),
        ):
            return None
        return int(image.value or 0), int(format_value.value)
        if function is None:
            raise FilamentBridgeError("Filament depth swapchain ABI is unavailable")
        array_type = ctypes.c_void_p * len(values)
        self._check_result(
            function(
                self._handle,
                int(eye_index),
                array_type(*values),
                len(values),
                int(format),
                int(width),
                int(height),
                ctypes.c_void_p(_as_pointer_value(depth_image)),
                int(depth_format),
            ),
            "create_eye_swapchain_with_depth",
        )

    def create_stereo_swapchain(
        self,
        image_handles: Iterable[Any],
        *,
        format: int,
        width: int,
        height: int,
    ) -> None:
        self._ensure_loaded()
        if not self.multiview_supported:
            raise FilamentBridgeError("Filament Vulkan multiview is unavailable")
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if not values:
            raise ValueError("Filament stereo swapchain requires at least one VkImage")
        array_type = ctypes.c_void_p * len(values)
        self._check_result(
            self._library.filament_bridge_create_stereo_swapchain(
                self._handle,
                array_type(*values),
                len(values),
                int(format),
                int(width),
                int(height),
            ),
            "create_stereo_swapchain",
        )

    def create_stereo_swapchain_with_depth(
        self,
        image_handles: Iterable[Any],
        *,
        format: int,
        width: int,
        height: int,
        depth_image: Any,
        depth_format: int,
    ) -> None:
        self._ensure_loaded()
        if not self.multiview_depth_swapchain_abi_available:
            raise FilamentBridgeError(
                "Filament Vulkan multiview depth swapchain is unavailable"
            )
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if not values:
            raise ValueError("Filament stereo swapchain requires at least one VkImage")
        array_type = ctypes.c_void_p * len(values)
        self._check_result(
            self._library.filament_bridge_create_stereo_swapchain_with_depth(
                self._handle,
                array_type(*values),
                len(values),
                int(format),
                int(width),
                int(height),
                ctypes.c_void_p(_as_pointer_value(depth_image)),
                int(depth_format),
            ),
            "create_stereo_swapchain_with_depth",
        )

    def create_controller_overlay_stereo_swapchain(
        self, image_handles: Iterable[Any], *, format: int,
        width: int, height: int, depth_image: Any, depth_format: int,
    ) -> None:
        self._ensure_loaded()
        if not self._controller_composition_layer_abi_available:
            raise FilamentBridgeError(
                "Filament controller composition layer ABI is unavailable"
            )
        values = [ctypes.c_void_p(_as_pointer_value(image)) for image in image_handles]
        if not values:
            raise ValueError("controller overlay swapchain requires VkImages")
        if depth_image is None:
            raise ValueError("controller overlay depth image must not be empty")
        array_type = ctypes.c_void_p * len(values)
        self._check_result(
            self._library.filament_bridge_create_controller_overlay_stereo_swapchain_with_depth(
                self._handle, array_type(*values), len(values), int(format),
                int(width), int(height),
                ctypes.c_void_p(_as_pointer_value(depth_image)),
                int(depth_format),
            ),
            "create_controller_overlay_stereo_swapchain",
        )

    def render_controller_composition_layer(self, image_index: int) -> None:
        self._ensure_loaded()
        if not self._controller_composition_layer_abi_available:
            raise FilamentBridgeError(
                "Filament controller composition layer ABI is unavailable"
            )
        self._check_result(
            self._library.filament_bridge_set_controller_overlay_acquired_image(
                self._handle, int(image_index)
            ),
            "set_controller_overlay_acquired_image",
        )
        self._check_result(
            self._library.filament_bridge_render_controller_composition_layer(
                self._handle
            ),
            "render_controller_composition_layer",
        )

    def set_active_eye(self, eye_index: int) -> None:
        self._ensure_loaded()
        if int(eye_index) not in (0, 1):
            raise ValueError("eye_index must be 0 or 1")
        self._check_result(
            self._library.filament_bridge_set_active_eye(self._handle, int(eye_index)),
            "set_active_eye",
        )

    def set_acquired_image(self, image_index: int) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_acquired_image(
                self._handle, int(image_index)
            ),
            "set_acquired_image",
        )

    @property
    def image_ready_semaphore_abi_available(self) -> bool:
        return self._image_ready_semaphore_abi_available

    def set_image_ready_semaphore(self, semaphore: int) -> None:
        self._ensure_loaded()
        if not self._image_ready_semaphore_abi_available:
            raise FilamentBridgeError("image-ready semaphore ABI is unavailable")
        self._check_result(
            self._library.filament_bridge_set_image_ready_semaphore(
                self._handle, ctypes.c_void_p(int(semaphore))
            ),
            "set_image_ready_semaphore",
        )

    def set_camera_look_at(
        self,
        eye: tuple[float, float, float],
        center: tuple[float, float, float],
        up: tuple[float, float, float],
    ) -> None:
        self._ensure_loaded()
        values = tuple(float(value) for value in (*eye, *center, *up))
        self._check_result(
            self._library.filament_bridge_set_camera_look_at(
                self._handle, *values
            ),
            "set_camera_look_at",
        )

    def set_camera_projection(
        self,
        vertical_fov_degrees: float,
        aspect: float,
        *,
        near_plane: float = 0.05,
        far_plane: float = 1000.0,
    ) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_camera_projection(
                self._handle,
                float(vertical_fov_degrees),
                float(aspect),
                float(near_plane),
                float(far_plane),
            ),
            "set_camera_projection",
        )

    def set_camera_projection_frustum(
        self,
        left: float,
        right: float,
        bottom: float,
        top: float,
        *,
        near_plane: float = 0.05,
        far_plane: float = 1000.0,
    ) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_camera_projection_frustum(
                self._handle,
                float(left), float(right), float(bottom), float(top),
                float(near_plane), float(far_plane),
            ),
            "set_camera_projection_frustum",
        )

    def set_stereo_camera(
        self,
        eye_model_matrices: Iterable[float],
        eye_frustums: Iterable[float],
        *,
        near_plane: float = 0.05,
        far_plane: float = 1000.0,
    ) -> None:
        self._ensure_loaded()
        if not self.multiview_abi_available:
            raise FilamentBridgeError("Filament Vulkan multiview ABI is unavailable")
        matrices = tuple(float(value) for value in eye_model_matrices)
        frustums = tuple(float(value) for value in eye_frustums)
        if len(matrices) != 32 or len(frustums) != 8:
            raise ValueError("stereo camera requires 32 matrix and 8 frustum values")
        self._check_result(
            self._library.filament_bridge_set_stereo_camera(
                self._handle,
                (ctypes.c_float * 32)(*matrices),
                (ctypes.c_double * 8)(*frustums),
                float(near_plane),
                float(far_plane),
            ),
            "set_stereo_camera",
        )

    def begin_frame(self) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_begin_frame(self._handle), "begin_frame"
        )

    def begin_background_frame(self) -> None:
        self._ensure_loaded()
        if not self._background_frame_abi_available:
            raise FilamentBridgeError("background frame ABI is unavailable")
        self._check_result(
            self._library.filament_bridge_begin_background_frame(self._handle),
            "begin_background_frame",
        )

    def render_controller_overlay(self) -> None:
        self._ensure_loaded()
        if not self._controller_overlay_abi_available:
            raise FilamentBridgeError("controller overlay ABI is unavailable")
        self._check_result(
            self._library.filament_bridge_render_controller_overlay(self._handle),
            "render_controller_overlay",
        )

    def render_controller_overlay_deferred(self) -> None:
        self._ensure_loaded()
        if not self._controller_overlay_batch_abi_available:
            raise FilamentBridgeError("deferred controller overlay ABI is unavailable")
        self._check_result(
            self._library.filament_bridge_render_controller_overlay_deferred(
                self._handle
            ),
            "render_controller_overlay_deferred",
        )

    def end_frame(self) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_end_frame(self._handle), "end_frame"
        )

    def end_frame_deferred(self) -> None:
        self._ensure_loaded()
        if not self._stereo_batch_submit_abi_available:
            raise FilamentBridgeError("stereo batch submit ABI is unavailable")
        self._check_result(
            self._library.filament_bridge_end_frame_deferred(self._handle),
            "end_frame_deferred",
        )

    def finish_frame_batch(self) -> None:
        self._ensure_loaded()
        if not self._stereo_batch_submit_abi_available:
            raise FilamentBridgeError("stereo batch submit ABI is unavailable")
        self._check_result(
            self._library.filament_bridge_finish_frame_batch(self._handle),
            "finish_frame_batch",
        )

    def wait_for_idle(self) -> None:
        self._ensure_loaded()
        if not self._async_submit_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_wait_for_idle(self._handle),
            "wait_for_idle",
        )

    def load_glb(self, data: bytes | bytearray | memoryview) -> None:
        self._ensure_loaded()
        payload = bytes(data)
        if not payload:
            raise ValueError("GLB payload must not be empty")
        payload, converted_materials = _prepare_environment_glb_for_dynamic_lighting(
            payload
        )
        if converted_materials:
            print(
                "[FilamentBridge] Environment dynamic-light materials enabled: "
                f"converted={len(converted_materials)}",
                flush=True,
            )
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_load_glb(
                self._handle, buffer, len(payload)
            ),
            "load_glb",
        )

    def unload_glb(self) -> None:
        self._ensure_loaded()
        function = getattr(self._library, "filament_bridge_unload_glb", None)
        if function is None:
            raise FilamentBridgeError(
                "Filament environment unload ABI is unavailable; update the Bridge DLL"
            )
        self._check_result(function(self._handle), "unload_glb")

    def load_controller(self, hand: int, data: bytes | bytearray | memoryview) -> None:
        self._ensure_loaded()
        self._ensure_controller_abi()
        payload = bytes(data)
        if int(hand) not in (0, 1) or not payload:
            raise ValueError("controller hand must be 0 or 1 and payload must not be empty")
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_load_controller(
                self._handle, int(hand), buffer, len(payload)
            ),
            "load_controller",
        )

    def set_controller_pose(self, hand: int, matrix) -> None:
        self._ensure_loaded()
        self._ensure_controller_abi()
        values = [float(value) for value in matrix.reshape(-1, order="F")]
        if int(hand) not in (0, 1) or len(values) != 16:
            raise ValueError("controller pose must be a 4x4 matrix")
        array_type = ctypes.c_float * 16
        self._check_result(
            self._library.filament_bridge_set_controller_pose(
                self._handle, int(hand), array_type(*values)
            ),
            "set_controller_pose",
        )

    def set_controller_inputs(
        self,
        hand: int,
        *,
        trigger: float,
        grip: float,
        joystick_x: float,
        joystick_y: float,
        button_mask: int,
    ) -> None:
        self._ensure_loaded()
        self._ensure_controller_abi()
        self._check_result(
            self._library.filament_bridge_set_controller_inputs(
                self._handle,
                int(hand),
                float(trigger),
                float(grip),
                float(joystick_x),
                float(joystick_y),
                int(button_mask),
            ),
            "set_controller_inputs",
        )

    def set_controller_visible(self, hand: int, visible: bool) -> None:
        self._ensure_loaded()
        if not self._controller_visibility_abi_available:
            return
        self._ensure_controller_abi()
        self._check_result(
            self._library.filament_bridge_set_controller_visible(
                self._handle, int(hand), int(bool(visible))
            ),
            "set_controller_visible",
        )

    def set_controller_material_override(
        self,
        *,
        roughness_factor: float | None,
        metallic_factor: float | None,
        specular_color_factor: tuple[float, float, float] | None,
    ) -> bool:
        self._ensure_loaded()
        if not self.controller_material_override_abi_available:
            return False
        specular = specular_color_factor or (-1.0, -1.0, -1.0)
        self._check_result(
            self._library.filament_bridge_set_controller_material_override(
                self._handle,
                -1.0 if roughness_factor is None else float(roughness_factor),
                -1.0 if metallic_factor is None else float(metallic_factor),
                *(float(value) for value in specular),
            ),
            "set_controller_material_override",
        )
        return True

    def set_controller_laser(self, hand: int, matrix, *, visible: bool) -> None:
        self._ensure_loaded()
        if not self._laser_abi_available:
            return
        self._ensure_controller_abi()
        values = [float(value) for value in matrix.reshape(-1, order="F")]
        if int(hand) not in (0, 1) or len(values) != 16:
            raise ValueError("controller laser matrix must be 4x4")
        array_type = ctypes.c_float * 16
        self._check_result(
            self._library.filament_bridge_set_controller_laser(
                self._handle, int(hand), array_type(*values), int(bool(visible))
            ),
            "set_controller_laser",
        )

    def set_controller_guide_texture(self, rgba) -> None:
        self._ensure_loaded()
        if not self._controller_guide_abi_available:
            return
        if getattr(rgba, "ndim", 0) != 3 or rgba.shape[2] != 4:
            raise ValueError("controller guide texture must be HxWx4 RGBA")
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        payload = bytes(memoryview(rgba).cast("B"))
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_set_controller_guide_texture(
                self._handle, buffer, width, height
            ),
            "set_controller_guide_texture",
        )

    def set_controller_guide(self, matrix, *, visible: bool) -> None:
        self._ensure_loaded()
        if not self._controller_guide_abi_available:
            return
        values = [float(value) for value in matrix.reshape(-1, order="F")]
        if len(values) != 16:
            raise ValueError("controller guide pose must be a 4x4 matrix")
        array_type = ctypes.c_float * 16
        self._check_result(
            self._library.filament_bridge_set_controller_guide(
                self._handle, array_type(*values), int(bool(visible))
            ),
            "set_controller_guide",
        )

    def set_text_overlay_page_texture(self, page: int, rgba) -> None:
        """Upload one MSDF atlas page; the native Bridge owns the GPU copy."""
        self._ensure_loaded()
        if not self._text_overlay_abi_available:
            return
        if int(page) < 0 or int(page) >= 4:
            raise ValueError("MSDF atlas page must be in the range 0..3")
        if getattr(rgba, "ndim", 0) != 3 or rgba.shape[2] != 4:
            raise ValueError("MSDF atlas page must be HxWx4 RGBA")
        height, width = int(rgba.shape[0]), int(rgba.shape[1])
        payload = _array_payload(rgba)
        buffer = ctypes.create_string_buffer(payload)
        self._check_result(
            self._library.filament_bridge_set_text_overlay_page_texture(
                self._handle, int(page), buffer, width, height
            ),
            "set_text_overlay_page_texture",
        )

    def set_text_overlay_page(
        self, page: int, vertices, indices, *, visible: bool
    ) -> None:
        """Submit packed world-space MSDF vertices on the Presenter thread."""
        self._ensure_loaded()
        if not self._text_overlay_abi_available:
            return
        if getattr(vertices, "ndim", 0) != 2 or vertices.shape[1] != 9:
            raise ValueError("MSDF vertices must have shape (N, 9)")
        if getattr(indices, "ndim", 0) != 1:
            raise ValueError("MSDF indices must be a one-dimensional array")
        if getattr(vertices, "itemsize", 0) != ctypes.sizeof(ctypes.c_float):
            raise ValueError("MSDF vertices must use float32 storage")
        if getattr(indices, "itemsize", 0) != ctypes.sizeof(ctypes.c_uint16):
            raise ValueError("MSDF indices must use uint16 storage")
        if hasattr(vertices, "flags") and not vertices.flags.c_contiguous:
            raise ValueError("MSDF vertices must be contiguous")
        if hasattr(indices, "flags") and not indices.flags.c_contiguous:
            raise ValueError("MSDF indices must be contiguous")
        vertex_payload = _array_payload(vertices)
        index_payload = _array_payload(indices)
        vertex_buffer = ctypes.create_string_buffer(vertex_payload)
        index_buffer = ctypes.create_string_buffer(index_payload)
        self._check_result(
            self._library.filament_bridge_set_text_overlay_page_vertices(
                self._handle,
                int(page),
                vertex_buffer,
                int(vertices.shape[0]),
                index_buffer,
                int(indices.shape[0]),
                int(bool(visible)),
            ),
            "set_text_overlay_page",
        )

    def set_scene_exposure(self, exposure_ev: float) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_scene_exposure(
                self._handle, float(exposure_ev)
            ),
            "set_scene_exposure",
        )

    def set_skybox_brightness(self, brightness: float) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_skybox_brightness(
                self._handle, float(brightness)
            ),
            "set_skybox_brightness",
        )

    def set_passthrough_backdrop(self, enabled: bool) -> None:
        self._ensure_loaded()
        if not self._passthrough_backdrop_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge passthrough-backdrop ABI is unavailable; rebuild the CI artifact"
            )
        self._check_result(
            self._library.filament_bridge_set_passthrough_backdrop(
                self._handle, int(bool(enabled))
            ),
            "set_passthrough_backdrop",
        )

    def set_fill_light(self, color, intensity: float, direction) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_set_fill_light(
                self._handle,
                *(float(value) for value in (*color, intensity, *direction)),
            ),
            "set_fill_light",
        )

    def set_ambient_light(self, color) -> None:
        self._ensure_loaded()
        if not self._ambient_light_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_set_ambient_light(
                self._handle, *(float(value) for value in color)
            ),
            "set_ambient_light",
        )

    def set_controller_ambient_light(self, color, enabled: bool) -> None:
        self._ensure_loaded()
        if not self._controller_ambient_light_abi_available:
            return
        self._check_result(
            self._library.filament_bridge_set_controller_ambient_light(
                self._handle,
                *(float(value) for value in color),
                int(bool(enabled)),
            ),
            "set_controller_ambient_light",
        )

    @property
    def lighting_config_abi_available(self) -> bool:
        return self._lighting_config_abi_available

    def set_lighting_config(
        self, *, environment_ambient_color, environment_ambient_intensity_lux,
        controller_ambient_color, controller_ambient_intensity_lux,
        controller_ambient_enabled, head_light_color,
        head_light_intensity_candela, head_light_offset, head_light_falloff,
        head_light_cast_shadows, top_light_color,
        top_light_intensity_candela, top_light_offset, top_light_falloff,
        top_light_cast_shadows,
    ) -> bool:
        self._ensure_loaded()
        if not self._lighting_config_abi_available:
            return False
        config = _FilamentLightingConfig(
            struct_size=ctypes.sizeof(_FilamentLightingConfig),
            environment_ambient_color=(ctypes.c_float * 3)(
                *(float(value) for value in environment_ambient_color)
            ),
            environment_ambient_intensity_lux=float(environment_ambient_intensity_lux),
            controller_ambient_color=(ctypes.c_float * 3)(
                *(float(value) for value in controller_ambient_color)
            ),
            controller_ambient_intensity_lux=float(controller_ambient_intensity_lux),
            controller_ambient_enabled=int(bool(controller_ambient_enabled)),
            head_light_color=(ctypes.c_float * 3)(
                *(float(value) for value in head_light_color)
            ),
            head_light_intensity_candela=float(head_light_intensity_candela),
            head_light_offset=(ctypes.c_float * 3)(
                *(float(value) for value in head_light_offset)
            ),
            head_light_falloff=float(head_light_falloff),
            head_light_cast_shadows=int(bool(head_light_cast_shadows)),
            top_light_color=(ctypes.c_float * 3)(
                *(float(value) for value in top_light_color)
            ),
            top_light_intensity_candela=float(top_light_intensity_candela),
            top_light_offset=(ctypes.c_float * 3)(
                *(float(value) for value in top_light_offset)
            ),
            top_light_falloff=float(top_light_falloff),
            top_light_cast_shadows=int(bool(top_light_cast_shadows)),
        )
        self._check_result(
            self._library.filament_bridge_set_lighting_config(
                self._handle, ctypes.byref(config)
            ),
            "set_lighting_config",
        )
        return True

    def set_controller_screen_light(
        self, color, intensity_lux: float, direction,
        cast_shadows: bool, enabled: bool,
    ) -> bool:
        self._ensure_loaded()
        if not self._controller_screen_light_abi_available:
            return False
        self._check_result(
            self._library.filament_bridge_set_controller_screen_light(
                self._handle,
                *(float(value) for value in color),
                float(intensity_lux),
                *(float(value) for value in direction),
                int(bool(cast_shadows)),
                int(bool(enabled)),
            ),
            "set_controller_screen_light",
        )
        return True

    def set_environment_screen_lights(
        self, positions, colors, intensities, *, falloff: float,
        cast_shadows: bool, enabled: bool,
    ) -> bool:
        self._ensure_loaded()
        if not self._environment_screen_lights_abi_available:
            return False
        position_values = tuple(
            float(value) for position in positions for value in position
        )
        color_values = tuple(
            float(value) for color in colors for value in color
        )
        intensity_values = tuple(float(value) for value in intensities)
        count = len(intensity_values)
        if len(position_values) != count * 3 or len(color_values) != count * 3:
            raise ValueError("environment screen light arrays have different sizes")
        if count > 24:
            raise ValueError("environment screen light count exceeds 24")
        float_array = ctypes.c_float * max(1, count * 3)
        intensity_array = ctypes.c_float * max(1, count)
        self._check_result(
            self._library.filament_bridge_set_environment_screen_lights(
                self._handle,
                float_array(*position_values) if count else float_array(),
                float_array(*color_values) if count else float_array(),
                intensity_array(*intensity_values) if count else intensity_array(),
                count,
                float(falloff),
                int(bool(cast_shadows)),
                int(bool(enabled)),
            ),
            "set_environment_screen_lights",
        )
        return True

    def set_environment_screen_area_light(
        self, position, normal, color, *, half_width: float,
        half_height: float, intensity: float, enabled: bool,
    ) -> bool:
        self._ensure_loaded()
        if not self._environment_screen_area_light_abi_available:
            return False
        vector = ctypes.c_float * 3
        self._check_result(
            self._library.filament_bridge_set_environment_screen_area_light(
                self._handle,
                vector(*(float(value) for value in position)),
                vector(*(float(value) for value in normal)),
                vector(*(float(value) for value in color)),
                float(half_width), float(half_height), float(intensity),
                int(bool(enabled)),
            ),
            "set_environment_screen_area_light",
        )
        return True

    @property
    def finished_drawing_semaphore_abi_available(self) -> bool:
        return self._finished_drawing_semaphore_abi_available

    def get_finished_drawing_semaphore(self) -> int | None:
        self._ensure_loaded()
        if not self._finished_drawing_semaphore_abi_available:
            return None
        output = ctypes.c_void_p()
        result = self._library.filament_bridge_get_finished_drawing_semaphore(
            self._handle, ctypes.byref(output)
        )
        if int(result) == 0 or not output.value:
            return None
        return int(output.value)

    def apply_animations(self, time_seconds: float) -> None:
        self._ensure_loaded()
        self._check_result(
            self._library.filament_bridge_apply_animations(
                self._handle, float(time_seconds)
            ),
            "apply_animations",
        )

    def close(self) -> None:
        if self._handle is not None:
            self._ensure_owner_thread()
            self._library.filament_bridge_destroy(self._handle)
            self._handle = None
        self._owner_thread_id = None

    def __enter__(self) -> "FilamentVulkanBridge":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _configure_abi(self) -> None:
        library = self._library
        library.filament_bridge_create_vulkan.argtypes = [
            ctypes.POINTER(_VulkanCreateInfo)
        ]
        library.filament_bridge_create_vulkan.restype = ctypes.c_void_p
        library.filament_bridge_destroy.argtypes = [ctypes.c_void_p]
        library.filament_bridge_destroy.restype = None
        library.filament_bridge_create_swapchain.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        library.filament_bridge_create_swapchain.restype = ctypes.c_int
        library.filament_bridge_create_eye_swapchain.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        library.filament_bridge_create_eye_swapchain.restype = ctypes.c_int
        create_eye_swapchain_with_depth = getattr(
            library, "filament_bridge_create_eye_swapchain_with_depth", None
        )
        if create_eye_swapchain_with_depth is not None:
            create_eye_swapchain_with_depth.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint32,
                ctypes.c_int32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_int32,
            ]
            create_eye_swapchain_with_depth.restype = ctypes.c_int
            self._depth_swapchain_abi_available = True
        multiview_abi = getattr(
            library, "filament_bridge_multiview_abi_available", None
        )
        multiview_supported = getattr(
            library, "filament_bridge_multiview_supported", None
        )
        create_stereo_swapchain = getattr(
            library, "filament_bridge_create_stereo_swapchain", None
        )
        create_stereo_swapchain_with_depth = getattr(
            library, "filament_bridge_create_stereo_swapchain_with_depth", None
        )
        create_controller_overlay_swapchain = getattr(
            library,
            "filament_bridge_create_controller_overlay_stereo_swapchain_with_depth",
            None,
        )
        set_controller_overlay_image = getattr(
            library,
            "filament_bridge_set_controller_overlay_acquired_image",
            None,
        )
        render_controller_composition_layer = getattr(
            library,
            "filament_bridge_render_controller_composition_layer",
            None,
        )
        set_stereo_camera = getattr(
            library, "filament_bridge_set_stereo_camera", None
        )
        if all(
            function is not None
            for function in (
                multiview_abi,
                multiview_supported,
                create_stereo_swapchain,
                set_stereo_camera,
            )
        ):
            multiview_abi.argtypes = []
            multiview_abi.restype = ctypes.c_int
            multiview_supported.argtypes = [ctypes.c_void_p]
            multiview_supported.restype = ctypes.c_int
            create_stereo_swapchain.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint32,
                ctypes.c_int32,
                ctypes.c_uint32,
                ctypes.c_uint32,
            ]
            create_stereo_swapchain.restype = ctypes.c_int
            if create_stereo_swapchain_with_depth is not None:
                create_stereo_swapchain_with_depth.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint32,
                    ctypes.c_int32,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                    ctypes.c_int32,
                ]
                create_stereo_swapchain_with_depth.restype = ctypes.c_int
                self._multiview_depth_swapchain_abi_available = True
            if all(
                function is not None
                for function in (
                    create_controller_overlay_swapchain,
                    set_controller_overlay_image,
                    render_controller_composition_layer,
                )
            ):
                create_controller_overlay_swapchain.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint32,
                    ctypes.c_int32,
                    ctypes.c_uint32,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                    ctypes.c_int32,
                ]
                create_controller_overlay_swapchain.restype = ctypes.c_int
                set_controller_overlay_image.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                ]
                set_controller_overlay_image.restype = ctypes.c_int
                render_controller_composition_layer.argtypes = [ctypes.c_void_p]
                render_controller_composition_layer.restype = ctypes.c_int
                self._controller_composition_layer_abi_available = True
            set_stereo_camera.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_double,
            ]
            set_stereo_camera.restype = ctypes.c_int
            self._multiview_abi_version = int(multiview_abi())
        library.filament_bridge_set_active_eye.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_bridge_set_active_eye.restype = ctypes.c_int
        library.filament_bridge_set_acquired_image.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_bridge_set_acquired_image.restype = ctypes.c_int
        if hasattr(library, "filament_bridge_set_image_ready_semaphore"):
            library.filament_bridge_set_image_ready_semaphore.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p
            ]
            library.filament_bridge_set_image_ready_semaphore.restype = ctypes.c_int
            self._image_ready_semaphore_abi_available = True
        library.filament_bridge_set_camera_look_at.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        library.filament_bridge_set_camera_look_at.restype = ctypes.c_int
        library.filament_bridge_set_camera_projection.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
        ]
        library.filament_bridge_set_camera_projection.restype = ctypes.c_int
        library.filament_bridge_set_camera_projection_frustum.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double,
        ]
        library.filament_bridge_set_camera_projection_frustum.restype = ctypes.c_int
        for name in ("filament_bridge_begin_frame", "filament_bridge_end_frame"):
            function = getattr(library, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int
        controller_overlay = getattr(
            library, "filament_bridge_render_controller_overlay", None
        )
        if controller_overlay is not None:
            controller_overlay.argtypes = [ctypes.c_void_p]
            controller_overlay.restype = ctypes.c_int
            self._controller_overlay_abi_available = True
        deferred_controller_overlay = getattr(
            library, "filament_bridge_render_controller_overlay_deferred", None
        )
        background_frame = getattr(
            library, "filament_bridge_begin_background_frame", None
        )
        if background_frame is not None:
            background_frame.argtypes = [ctypes.c_void_p]
            background_frame.restype = ctypes.c_int
            self._background_frame_abi_available = True
        deferred_end_frame = getattr(
            library, "filament_bridge_end_frame_deferred", None
        )
        finish_frame_batch = getattr(
            library, "filament_bridge_finish_frame_batch", None
        )
        if deferred_end_frame is not None and finish_frame_batch is not None:
            deferred_end_frame.argtypes = [ctypes.c_void_p]
            deferred_end_frame.restype = ctypes.c_int
            finish_frame_batch.argtypes = [ctypes.c_void_p]
            finish_frame_batch.restype = ctypes.c_int
            self._stereo_batch_submit_abi_available = True
        if (
            deferred_controller_overlay is not None
            and self._stereo_batch_submit_abi_available
        ):
            deferred_controller_overlay.argtypes = [ctypes.c_void_p]
            deferred_controller_overlay.restype = ctypes.c_int
            self._controller_overlay_batch_abi_available = True
        wait_for_idle = getattr(library, "filament_bridge_wait_for_idle", None)
        if wait_for_idle is not None:
            wait_for_idle.argtypes = [ctypes.c_void_p]
            wait_for_idle.restype = ctypes.c_int
            self._async_submit_abi_available = True
        library.filament_bridge_load_glb.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32
        ]
        library.filament_bridge_load_glb.restype = ctypes.c_int
        unload_glb = getattr(library, "filament_bridge_unload_glb", None)
        if unload_glb is not None:
            unload_glb.argtypes = [ctypes.c_void_p]
            unload_glb.restype = ctypes.c_int
            self._environment_unload_abi_available = True
        controller_functions = (
            "filament_bridge_load_controller",
            "filament_bridge_set_controller_pose",
            "filament_bridge_set_controller_inputs",
        )
        if all(hasattr(library, name) for name in controller_functions):
            library.filament_bridge_load_controller.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32
            ]
            library.filament_bridge_load_controller.restype = ctypes.c_int
            library.filament_bridge_set_controller_pose.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)
            ]
            library.filament_bridge_set_controller_pose.restype = ctypes.c_int
            library.filament_bridge_set_controller_inputs.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint32,
            ]
            library.filament_bridge_set_controller_inputs.restype = ctypes.c_int
            self._controller_abi_available = True
        set_controller_visible = getattr(
            library, "filament_bridge_set_controller_visible", None
        )
        if set_controller_visible is not None:
            set_controller_visible.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int
            ]
            set_controller_visible.restype = ctypes.c_int
            self._controller_visibility_abi_available = True
        set_controller_material_override = getattr(
            library, "filament_bridge_set_controller_material_override", None
        )
        if set_controller_material_override is not None:
            set_controller_material_override.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
            ]
            set_controller_material_override.restype = ctypes.c_int
            self._controller_material_override_abi_available = True
        set_controller_laser = getattr(
            library, "filament_bridge_set_controller_laser", None
        )
        if set_controller_laser is not None:
            set_controller_laser.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_int
            ]
            set_controller_laser.restype = ctypes.c_int
            self._laser_abi_available = True
        guide_texture = getattr(
            library, "filament_bridge_set_controller_guide_texture", None
        )
        guide_pose = getattr(library, "filament_bridge_set_controller_guide", None)
        if guide_texture is not None and guide_pose is not None:
            guide_texture.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32,
            ]
            guide_texture.restype = ctypes.c_int
            guide_pose.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int
            ]
            guide_pose.restype = ctypes.c_int
            self._controller_guide_abi_available = True
        text_texture = getattr(
            library, "filament_bridge_set_text_overlay_page_texture", None
        )
        text_vertices = getattr(
            library, "filament_bridge_set_text_overlay_page_vertices", None
        )
        if text_texture is not None and text_vertices is not None:
            text_texture.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32,
            ]
            text_texture.restype = ctypes.c_int
            text_vertices.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_int,
            ]
            text_vertices.restype = ctypes.c_int
            self._text_overlay_abi_available = True
        library.filament_bridge_set_scene_exposure.argtypes = [
            ctypes.c_void_p, ctypes.c_float
        ]
        library.filament_bridge_set_scene_exposure.restype = ctypes.c_int
        library.filament_bridge_set_skybox_brightness.argtypes = [
            ctypes.c_void_p, ctypes.c_float
        ]
        library.filament_bridge_set_skybox_brightness.restype = ctypes.c_int
        set_passthrough_backdrop = getattr(
            library, "filament_bridge_set_passthrough_backdrop", None
        )
        if set_passthrough_backdrop is not None:
            set_passthrough_backdrop.argtypes = [ctypes.c_void_p, ctypes.c_int]
            set_passthrough_backdrop.restype = ctypes.c_int
            self._passthrough_backdrop_abi_available = True
        set_ambient_light = getattr(
            library, "filament_bridge_set_ambient_light", None
        )
        if set_ambient_light is not None:
            set_ambient_light.argtypes = [
                ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float
            ]
            set_ambient_light.restype = ctypes.c_int
            self._ambient_light_abi_available = True
        set_controller_ambient_light = getattr(
            library, "filament_bridge_set_controller_ambient_light", None
        )
        if set_controller_ambient_light is not None:
            set_controller_ambient_light.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int,
            ]
            set_controller_ambient_light.restype = ctypes.c_int
            self._controller_ambient_light_abi_available = True
        set_lighting_config = getattr(
            library, "filament_bridge_set_lighting_config", None
        )
        if set_lighting_config is not None:
            set_lighting_config.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(_FilamentLightingConfig)
            ]
            set_lighting_config.restype = ctypes.c_int
            self._lighting_config_abi_available = True
        set_controller_screen_light = getattr(
            library, "filament_bridge_set_controller_screen_light", None
        )
        if set_controller_screen_light is not None:
            set_controller_screen_light.argtypes = [
                ctypes.c_void_p,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int,
            ]
            set_controller_screen_light.restype = ctypes.c_int
            self._controller_screen_light_abi_available = True
        set_environment_screen_lights = getattr(
            library, "filament_bridge_set_environment_screen_lights", None
        )
        if set_environment_screen_lights is not None:
            set_environment_screen_lights.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint32,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
            ]
            set_environment_screen_lights.restype = ctypes.c_int
            self._environment_screen_lights_abi_available = True
        set_environment_screen_area_light = getattr(
            library, "filament_bridge_set_environment_screen_area_light", None
        )
        if set_environment_screen_area_light is not None:
            set_environment_screen_area_light.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
            ]
            set_environment_screen_area_light.restype = ctypes.c_int
            self._environment_screen_area_light_abi_available = True
        library.filament_bridge_set_fill_light.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        library.filament_bridge_set_fill_light.restype = ctypes.c_int
        external_image_abi = getattr(
            library, "filament_bridge_vulkan_external_image_abi_available", None
        )
        if external_image_abi is not None:
            external_image_abi.argtypes = [ctypes.c_void_p]
            external_image_abi.restype = ctypes.c_int
            self._vulkan_external_image_abi_available = bool(
                external_image_abi(None)
            )
        depth_output_abi = getattr(
            library, "filament_bridge_depth_output_abi_available", None
        )
        if depth_output_abi is not None:
            depth_output_abi.argtypes = [ctypes.c_void_p]
            depth_output_abi.restype = ctypes.c_int
            self._depth_output_abi_available = bool(depth_output_abi(None))
        depth_attachment = getattr(
            library, "filament_bridge_get_depth_attachment", None
        )
        if depth_attachment is not None:
            depth_attachment.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int32),
            ]
            depth_attachment.restype = ctypes.c_int
        if hasattr(library, "filament_bridge_get_finished_drawing_semaphore"):
            library.filament_bridge_get_finished_drawing_semaphore.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            ]
            library.filament_bridge_get_finished_drawing_semaphore.restype = ctypes.c_int
            self._finished_drawing_semaphore_abi_available = True
        library.filament_bridge_apply_animations.argtypes = [
            ctypes.c_void_p, ctypes.c_double
        ]
        library.filament_bridge_apply_animations.restype = ctypes.c_int
        library.filament_bridge_last_error.argtypes = [ctypes.c_void_p]
        library.filament_bridge_last_error.restype = ctypes.c_char_p

    def _ensure_controller_abi(self) -> None:
        if not self._controller_abi_available:
            raise FilamentBridgeError(
                "Filament Bridge controller ABI is unavailable; rebuild the CI artifact"
            )

    def _ensure_loaded(self) -> None:
        if not self.loaded:
            raise FilamentBridgeError("Filament Bridge is not initialized")
        self._ensure_owner_thread()

    def _ensure_owner_thread(self) -> None:
        if (
            self._owner_thread_id is not None
            and threading.get_ident() != self._owner_thread_id
        ):
            raise FilamentBridgeError(
                "Filament Bridge C ABI must run on its Presenter owner thread"
            )

    def _raise_if_error(self) -> None:
        message = self._last_error()
        if message:
            self.close()
            raise FilamentBridgeError(f"create_vulkan: {message}")

    def _check_result(self, result: int, operation: str) -> None:
        if int(result) == 0:
            message = self._last_error() or "Filament Bridge operation failed"
            raise FilamentBridgeError(f"{operation}: {message}")

    def _last_error(self) -> str:
        value = self._library.filament_bridge_last_error(self._handle)
        return value.decode("utf-8", errors="replace") if value else ""


def _as_pointer_value(value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, ctypes.c_void_p):
        return int(value.value or 0)
    try:
        import vulkan as vk

        return int(vk.ffi.cast("uintptr_t", value))
    except (ImportError, TypeError, ValueError):
        return int(ctypes.cast(value, ctypes.c_void_p).value or 0)


def _array_payload(value: Any) -> bytes:
    """Serialize NumPy buffers safely, including zero-length arrays."""
    tobytes = getattr(value, "tobytes", None)
    if tobytes is not None:
        return bytes(tobytes(order="C"))
    return bytes(memoryview(value).cast("B"))
