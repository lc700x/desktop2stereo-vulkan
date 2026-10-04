from __future__ import annotations

import ctypes
import json
import re
import struct
import subprocess
import threading
from contextlib import ExitStack
from pathlib import Path

import pytest

from path_config import APP_ROOT
from xr_viewer.filament_vulkan_bridge import (
    FilamentBridgeError,
    FilamentVulkanBridge,
    _FilamentLightingConfig,
    _VulkanCreateInfo,
    _as_pointer_value,
    _prepare_environment_glb_for_dynamic_lighting,
    default_bridge_path,
)


def test_vulkan_create_info_has_stable_c_layout() -> None:
    assert ctypes.sizeof(_VulkanCreateInfo) == ctypes.sizeof(ctypes.c_void_p) * 3 + 8
    assert ctypes.sizeof(_FilamentLightingConfig) == 112


def test_default_bridge_path_matches_platform() -> None:
    path = default_bridge_path()
    assert path.parent.name in {"windows", "linux", "macos"}
    assert path.name.startswith(("filament_bridge", "libfilament_bridge"))


def _make_test_glb(document: dict, binary: bytes = b"\x00\x01\x02\x03") -> bytes:
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 4)
    binary += b"\x00" * ((-len(binary)) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    return (
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
        + struct.pack("<II", len(binary), 0x004E4942) + binary
    )


def _read_test_glb_json(payload: bytes) -> dict:
    length, chunk_type = struct.unpack_from("<II", payload, 12)
    assert chunk_type == 0x4E4F534A
    return json.loads(payload[20:20 + length].decode("utf-8").rstrip("\x00 "))


def test_environment_glb_converts_room_surfaces_but_preserves_emissive_unlit() -> None:
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "materials": [
            {
                "name": "M_Wall_gltf_unlit",
                "extensions": {"KHR_materials_unlit": {}},
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 2},
                },
                "normalTexture": {"index": 3},
                "occlusionTexture": {"index": 4},
            },
            {
                "name": "M_fakelight_gltf_unlit",
                "extensions": {"KHR_materials_unlit": {}},
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}},
            },
            {
                "name": "Solid_emissive_color",
                "extensions": {"KHR_materials_unlit": {}},
                "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]},
            },
        ],
    }

    converted, names = _prepare_environment_glb_for_dynamic_lighting(
        _make_test_glb(document)
    )
    result = _read_test_glb_json(converted)

    assert names == ("M_Wall_gltf_unlit",)
    wall = result["materials"][0]
    assert "extensions" not in wall
    assert wall["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    assert wall["pbrMetallicRoughness"]["roughnessFactor"] == 0.85
    assert wall["emissiveTexture"] == {"index": 0}
    assert wall["emissiveFactor"] == [1.0, 1.0, 1.0]
    assert wall["name"] == "D2S_REFLECTIVE__M_Wall_gltf_unlit"
    assert "metallicRoughnessTexture" not in wall["pbrMetallicRoughness"]
    assert "normalTexture" not in wall
    assert "occlusionTexture" not in wall
    assert "KHR_materials_unlit" in result["materials"][1]["extensions"]
    assert "KHR_materials_unlit" in result["materials"][2]["extensions"]
    assert result["extensionsUsed"] == ["KHR_materials_unlit"]


def test_environment_reflection_material_uses_continuous_screen_area_light() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "native/filament/bridge/bridge_material_provider.cpp"
    ).read_text(encoding="utf-8")
    context = (
        root / "native/filament/bridge/bridge_context.cpp"
    ).read_text(encoding="utf-8")

    assert 'constexpr char kReflectivePrefix[] = "D2S_REFLECTIVE__"' in source
    assert ".customSurfaceShading(true)" not in source
    assert "getMaterialGlobal0()" in source
    assert "getUserWorldPosition()" in source
    assert "screenAreaLight" in source
    assert "vec3 bakedBaseline = material.baseColor.rgb" in source
    assert "* materialParams.emissiveFactor" in source
    assert "distanceToScreen * distanceToScreen" in source
    assert "reflective room material active" in source
    assert "bridge_create_material_provider(bridge->engine)" in context


def test_small_theater_uses_shared_continuous_screen_area_light() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((
        APP_ROOT / "xr_viewer/environments/3d_theater/profile.json"
    ).read_text(encoding="utf-8"))

    assert profile["screen_light_intensity"] == pytest.approx(6.0)
    assert "environment_screen_light_surface_gain" not in profile
    assert "environment_screen_light_position_inset" not in profile


def test_remote_filament_build_enables_multiview_without_stale_sdk_cache() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github/workflows/filament-bridge.yml"
    ).read_text(encoding="utf-8")
    patch = (
        root / "native/filament/patches/apply_d2s_vulkan_external_image.py"
    ).read_text(encoding="utf-8")
    manifest = json.loads(
        (root / "native/filament/version.json").read_text(encoding="utf-8")
    )
    cmake = (root / "native/filament/bridge/CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    assert manifest["version"] == "1.75.0"
    assert manifest["source"]["ref"] == "v1.75.0"
    assert all("v1.75.0" in entry["asset"] for entry in manifest["platforms"].values())
    assert "google/filament/v1.75.0/libs/bluevk" in cmake
    assert "#if defined(VK_EXT_depth_clamp_control)" in cmake
    assert "VK_EXT_depth_clamp_control) || defined(VK_EXT_shader_object" in cmake
    assert workflow.count("-DFILAMENT_ENABLE_MULTIVIEW=ON") == 2
    assert workflow.count(
        "hashFiles('native/filament/version.json', "
        "'native/filament/patches/**', '.github/workflows/filament-bridge.yml')"
    ) == 2
    assert "[D2S stereo trace] renderer" in patch
    assert "d2sStereoTraceLogged[2]" in patch
    assert "variant.hasStereo()" in patch
    assert "[D2S stereo trace] renderPass" in patch
    assert "config.viewCount, subpassViewMask" in patch
    assert "[D2S stereo trace] program" in patch
    assert "builder.isMultiview()" in patch
    assert "words[word] == 4440u" in patch
    assert "fragmentViewIndex=%u" in patch
    assert "D2S_FILAMENT_SHADER_DUMP_DIR" in patch
    assert "filament_controller_eye_diag.%s.spv" in patch
    assert "[D2S stereo trace] controller draw program=%s viewCount=%u " in patch
    # Stereo projection is handled by Filament's generated multiview variants.
    # Do not override its global PBR camera helpers: doing so made PICO's
    # zero-roughness shell break into view-dependent reflective facets.
    assert "common_getters.glsl" not in patch
    assert "surface_shading_parameters.fs" not in patch
    assert "worldFromEye" not in patch
    assert 'std::strcmp(program->name.c_str(), "D2S Controller Eye Diagnostic") == 0' in patch
    assert "rt->getRenderPassKey().viewCount" in patch
    assert "rt->getRenderPassKey().needsResolveMask" in patch
    assert "rt->getFboKey().layers" in patch
    assert "d2sColor.texture->getPrimaryViewRange()" in patch
    assert "d2sColor.texture->getViewType()" in patch
    assert "getViewType(getSamplerTypeFromDepth(depth))" in patch
    assert "CLEARDEPTH_MULTIVIEW" in patch
    assert "engine.getConfig().stereoscopicType == StereoscopicType::MULTIVIEW" in patch


def test_pointer_value_accepts_integer_and_c_void_p() -> None:
    assert _as_pointer_value(17) == 17
    assert _as_pointer_value(ctypes.c_void_p(23)) == 23


def test_missing_bridge_library_is_reported() -> None:
    with pytest.raises(FilamentBridgeError, match="unable to load"):
        FilamentVulkanBridge("missing-filament-bridge.dll")


def test_bridge_rejects_c_abi_calls_from_non_owner_thread() -> None:
    bridge = object.__new__(FilamentVulkanBridge)
    bridge._handle = ctypes.c_void_p(1)
    bridge._owner_thread_id = threading.get_ident() + 1

    with pytest.raises(FilamentBridgeError, match="Presenter owner thread"):
        bridge._ensure_loaded()


def test_native_bridge_keeps_modular_resource_lifetimes_explicit() -> None:
    bridge_dir = (
        Path(__file__).resolve().parents[1] / "native/filament/bridge"
    )
    cmake = (bridge_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    facade = (bridge_dir / "filament_bridge.cpp").read_text(encoding="utf-8")
    header = (bridge_dir / "filament_bridge.h").read_text(encoding="utf-8")

    assert not (bridge_dir / "bridge_screen.cpp").exists()
    assert not (bridge_dir / "bridge_screen.h").exists()
    assert "bridge_screen.cpp" not in cmake
    assert "filament_bridge_create_screen" not in facade
    assert "filament_bridge_create_screen" not in header
    assert "filament_bridge_set_screen_image" not in facade
    assert "filament_bridge_set_screen_image" not in header
    assert "filament_bridge_depth_output_abi_available" in facade
    depth_capability = facade[facade.index("filament_bridge_depth_output_abi_available"):]
    assert "return 1;" in depth_capability[:depth_capability.index("int filament_bridge_get_depth_attachment")]
    assert "filament_bridge_create_eye_swapchain_with_depth" in facade
    assert "filament_bridge_create_stereo_swapchain_with_depth" in facade
    assert "filament_bridge_multiview_abi_available() { return 3; }" in facade
    assert "filament_bridge_unload_glb" in facade
    assert "filament_bridge_unload_glb" in header
    assert "bridge_scene_destroy(bridge);" in facade
    assert "bundle.depth = swapchain->depth" in (
        bridge_dir / "bridge_internal.h"
    ).read_text(encoding="utf-8")


def test_controller_material_override_is_not_enabled_by_default_profiles() -> None:
    root = Path(__file__).resolve().parents[1]
    facade = (root / "native/filament/bridge/filament_bridge.cpp").read_text(
        encoding="utf-8"
    )
    controller = (root / "native/filament/bridge/bridge_controller.cpp").read_text(
        encoding="utf-8"
    )
    assert "filament_bridge_set_controller_material_override" in facade
    assert 'hasParameter("roughnessFactor")' in controller
    assert 'hasParameter("metallicFactor")' in controller
    assert 'hasParameter("specularColorFactor")' in controller
    for brand in ("HP", "INDEX", "PICO", "QUEST", "VIVE", "YVR"):
        profile = json.loads(
            (APP_ROOT / f"xr_viewer/controllers/{brand}/profile.json").read_text(
                encoding="utf-8"
            )
        )["overrides"]
        if brand == "PICO":
            assert profile["material_roughness_factor"] == pytest.approx(0.3)
            assert profile["material_metallic_factor"] == pytest.approx(0.0)
            assert profile["material_specular_color_factor"] == pytest.approx(
                [1.0, 1.0, 1.0]
            )
            assert profile["controller_head_light_cast_shadows"] is True
            assert profile["controller_top_light_cast_shadows"] is True
        else:
            assert "material_roughness_factor" not in profile
            assert "material_metallic_factor" not in profile
            assert "material_specular_color_factor" not in profile

def test_legacy_filament_screen_texture_light_path_stays_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (APP_ROOT / "xr_viewer/core_openxr_vulkan.py").read_text(
        encoding="utf-8"
    )
    facade = (root / "native/filament/bridge/filament_bridge.cpp").read_text(
        encoding="utf-8"
    )
    assert "def _update_filament_screen_light" not in source
    assert "filament_bridge_set_screen_light" not in facade
    assert "filament_bridge_set_controller_screen_light" in facade


def test_artemis_controller_lighting_matches_legacy_head_light() -> None:
    root = Path(__file__).resolve().parents[1]
    profile_path = APP_ROOT / "xr_viewer/environments/3D_Artemis/profile.json"
    if not profile_path.is_file():
        profile_path = APP_ROOT / "xr_viewer/environments/Artemis/profile.json"
    profile = json.loads(
        profile_path.read_text(encoding="utf-8")
    )
    assert profile["env_head_light_color"] == [0.45, 0.45, 0.48]
    assert profile["env_ambient_color"] == [0.06, 0.05, 0.05]
    assert profile["preview_exposure"] == 0.0
    config = (APP_ROOT / "xr_viewer/core_openxr_vulkan.py").read_text(
        encoding="utf-8"
    )
    assert "filament_fill_light_intensity: float = 1.0" in config
    assert 'profile.get("env_head_light_color"' in config
    assert '"env_ambient_color", self._filament_ambient_light_color' in config
    assert "bridge.set_ambient_light(self._controller_ambient_light_color())" in config

    native_lighting = (root / "native/filament/bridge/bridge_material.cpp").read_text(
        encoding="utf-8"
    )
    assert "kControllerHeadLightWeight" not in native_lighting
    assert "kControllerTopLightWeight" not in native_lighting
    assert "kLegacyControllerCandelaScale" not in native_lighting
    assert "config->head_light_intensity_candela" in native_lighting
    assert "config->top_light_intensity_candela" in native_lighting
    assert "bridge_material_set_controller_screen_light" in native_lighting
    assert "bridge_material_set_environment_screen_lights" in native_lighting
    assert ".lightChannel(0, false).lightChannel(1, true)" in native_lighting
    assert ".lightChannel(0, true).lightChannel(1, false)" in native_lighting
    material_header = (root / "native/filament/bridge/bridge_material.h").read_text(
        encoding="utf-8"
    )
    assert "struct FilamentBridgeLightingConfig;" in material_header

    common = json.loads(
        (APP_ROOT / "xr_viewer/environments/common.json").read_text(encoding="utf-8")
    )["filament"]
    assert common["controller_head_light_weight"] == pytest.approx(0.85)
    assert common["controller_top_light_weight"] == pytest.approx(0.6)
    assert common["controller_head_light_offset"] == [0.0, 0.05, 0.0]
    assert common["controller_top_light_offset"] == [0.0, 0.45, -0.18]
    assert common["controller_screen_light_enabled"] is True
    assert common["controller_screen_light_intensity_lux"] == pytest.approx(500.0)
    assert common["controller_screen_light_sample_hz"] == pytest.approx(12.0)
    assert common["environment_screen_light_enabled"] is True
    assert common["environment_screen_light_intensity_candela"] == pytest.approx(120.0)
    assert common["environment_screen_light_sample_hz"] == pytest.approx(12.0)
    assert common["glow_sample_hz"] == pytest.approx(30.0)
    assert common["glow_smoothing_seconds"] == pytest.approx(0.10)


@pytest.mark.parametrize("brand", ("HP", "INDEX", "PICO", "QUEST", "VIVE", "YVR"))
@pytest.mark.parametrize("hand", ("left", "right"))
def test_packaged_controller_glb_animation_triplets_have_native_fallbacks(
    brand: str, hand: str
) -> None:
    path = APP_ROOT / "xr_viewer/controllers" / brand / f"{hand}.glb"
    payload = path.read_bytes()
    assert payload[:4] == b"glTF"
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    document = json.loads(
        payload[20 : 20 + json_length].decode("utf-8").rstrip("\x00 ")
    )
    names = {str(node.get("name") or "") for node in document["nodes"]}
    value_names = {
        name
        for name in names
        if name.endswith("_value")
        and all(
            name.removesuffix("_value") + suffix in names
            for suffix in ("_min", "_max")
        )
    }

    assert value_names
    assert any("trigger_pressed_value" in name for name in value_names)
    assert any("squeeze_pressed_value" in name for name in value_names)
    assert any("thumbstick_pressed_value" in name for name in value_names)
    bridge_source = (
        Path(__file__).resolve().parents[1]
        / "native/filament/bridge/bridge_controller.cpp"
    ).read_text(encoding="utf-8")
    assert all(f'"{value_name}"' in bridge_source for value_name in value_names)


def test_native_controller_animation_preserves_touch_semantics_without_abi_growth() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/filament/bridge/bridge_controller.cpp").read_text(
        encoding="utf-8"
    )
    internal = (root / "native/filament/bridge/bridge_internal.h").read_text(
        encoding="utf-8"
    )
    public_header = (root / "native/filament/bridge/filament_bridge.h").read_text(
        encoding="utf-8"
    )

    assert '"joystick_x_touched"' in source
    assert '"joystick_y_touched"' in source
    assert '"joystick_touched"' in source
    assert "controller.button_values[6] * controller.joystick_x" in source
    assert "controller.button_values[6] * controller.joystick_y" in source
    assert "std::array<float, 7> button_values{};" in internal
    assert "uint32_t button_mask" in public_header


def test_native_controller_multiview_eye_diagnostic_replaces_glb_materials() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/filament/bridge/bridge_controller.cpp").read_text(
        encoding="utf-8"
    )
    internal = (root / "native/filament/bridge/bridge_internal.h").read_text(
        encoding="utf-8"
    )

    assert "D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC" in source
    assert "variable_d2sEyeIndex.x < 0.5" in source
    assert "material.d2sEyeIndex = float4(float(getEyeIndex())" in source
    assert '.variable(filamat::MaterialBuilder::Variable::CUSTOM0, "d2sEyeIndex")' in source
    assert "material.worldPosition.x += 10000.0" not in source
    assert "eye0=red eye1=green" in source
    assert (
        ".stereoscopicType(filamat::MaterialBuilder::StereoscopicType::MULTIVIEW)"
        in source
    )
    assert "renderables.setMaterialInstanceAt(" in source
    assert "controller_eye_diagnostic_material_instance" in internal


def test_controller_eye_diagnostic_launcher_enables_backend_stereo_trace() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "scripts/run_windows_filament_multiview_controller_eye_diagnostic.ps1"
    ).read_text(encoding="utf-8")

    assert '$env:D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC = "1"' in launcher
    assert '$env:D2S_FILAMENT_EYE_DIAGNOSTIC = "1"' in launcher
    assert '$env:D2S_FILAMENT_SHADER_DUMP_DIR = $shaderDumpDir' in launcher
    assert 'commonLightingPath = Join-Path $repoRoot "src\\desktop2stereo\\xr_viewer\\environments\\common.json"' in launcher
    assert 'controller_head_light_weight' in launcher
    assert 'controller_top_light_weight' in launcher
    assert 'controller_screen_light_intensity_lux' in launcher
    assert "[D2S stereo trace]" in launcher


def test_formal_launcher_clears_filament_diagnostics() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "src/run_windows.bat").read_text(encoding="utf-8")

    for name in (
        "D2S_FILAMENT_CONTROLLER_EYE_DIAGNOSTIC",
        "D2S_FILAMENT_EYE_DIAGNOSTIC",
        "D2S_FILAMENT_MULTIVIEW_LAYER_READBACK",
        "D2S_FILAMENT_SHADER_DUMP_DIR",
        "D2S_FILAMENT_MULTIVIEW_PROJECTION_DIAGNOSTIC",
        "D2S_FILAMENT_PROJECTION_ONLY",
        "D2S_OPENXR_PROJECTION_ARRAY_EYE_DIAGNOSTIC",
        "D2S_OPENXR_SCREEN_QUAD_EYE_DIAGNOSTIC",
        "D2S_OPENXR_VULKAN_MULTIVIEW_EYE_DIAGNOSTIC",
        "D2S_VULKAN_PROJECTION_COMPOSER_EYE_DIAGNOSTIC",
    ):
        assert f'set "{name}="' in launcher


def test_native_multiview_restores_validated_fresh_depth_foreground_pass() -> None:
    root = Path(__file__).resolve().parents[1]
    eye_source = (root / "native/filament/bridge/bridge_eye.cpp").read_text(
        encoding="utf-8"
    )
    context_source = (root / "native/filament/bridge/bridge_context.cpp").read_text(
        encoding="utf-8"
    )

    assert "bridge_screen_prepare_frame" not in eye_source
    assert "bridge_screen_bind_stereo_textures" not in eye_source
    assert "bridge->renderer->render(bridge->view);" in eye_source
    assert "bridge->renderer->render(eye.foreground_view);" in eye_source
    assert "0x01u | 0x02u | 0x04u | (1u << kScreenLayerBase)" in eye_source
    assert "eye.foreground_view->setChannelDepthClearEnabled(2, true);" in context_source
    assert "filament_bridge_create_stereo_swapchain_with_depth" in (
        root / "native/filament/bridge/filament_bridge.cpp"
    ).read_text(encoding="utf-8")
    assert "filament::View::BlendMode::OPAQUE" in eye_source
    assert "filament::View::BlendMode::TRANSLUCENT" in eye_source
    assert "eye.foreground_view->setBlendMode(foreground_blend);" in eye_source


def test_controller_lights_remain_in_foreground_scene_for_multiview() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/filament/bridge/bridge_material.cpp").read_text(
        encoding="utf-8"
    )

    assert "bridge->foreground_scene->addEntity(bridge->fill_light)" in source
    assert "bridge->foreground_scene->addEntity(bridge->controller_top_light)" in source
    assert "bridge->foreground_scene->addEntity(bridge->controller_screen_light)" in source
    assert "bridge->multiview_active ? bridge->scene : bridge->foreground_scene" not in source


def test_native_controller_overlay_preserves_composer_color() -> None:
    root = Path(__file__).resolve().parents[1]
    eye_source = (root / "native/filament/bridge/bridge_eye.cpp").read_text(
        encoding="utf-8"
    )
    overlay = eye_source[eye_source.index("int bridge_eye_render_controller_overlay"):]
    overlay = overlay[:overlay.index("namespace {")]

    assert "overlay_options.clear = false;" in overlay
    assert "overlay_options.discard = false;" in overlay
    assert "render(eye.controller_view)" in overlay
    assert "render(eye.controller_guide_view)" not in overlay
    assert "render(bridge->view)" not in overlay
    assert "render(eye.foreground_view)" not in overlay

    context_source = (
        root / "native/filament/bridge/bridge_context.cpp"
    ).read_text(encoding="utf-8")
    assert "eye.controller_view->setVisibleLayers(0xff, 0x05);" in context_source
    guide_source = (
        root / "native/filament/bridge/bridge_controller_guide.cpp"
    ).read_text(encoding="utf-8")
    assert ".depthCulling(false)" in guide_source
    assert (
        "bridge->foreground_scene->addEntity(bridge->controller_guide_entity)"
        in guide_source
    )
    assert (
        "bridge->multiview_active ? bridge->scene : bridge->foreground_scene"
        not in guide_source
    )


def test_native_background_frame_defers_controller_layers_to_overlay() -> None:
    root = Path(__file__).resolve().parents[1]
    eye_source = (root / "native/filament/bridge/bridge_eye.cpp").read_text(
        encoding="utf-8"
    )
    facade = (root / "native/filament/bridge/filament_bridge.cpp").read_text(
        encoding="utf-8"
    )
    public_header = (root / "native/filament/bridge/filament_bridge.h").read_text(
        encoding="utf-8"
    )

    assert "bridge_eye_begin_frame_impl(bridge, true)" in eye_source
    assert "bridge_eye_begin_frame_impl(bridge, false)" in eye_source
    assert "render_controller_layers && !bridge->multiview_active" in eye_source
    assert "filament_bridge_begin_background_frame" in facade
    assert "filament_bridge_begin_background_frame" in public_header


def test_native_multiview_uses_final_controller_composition_layer() -> None:
    root = Path(__file__).resolve().parents[1]
    eye_source = (root / "native/filament/bridge/bridge_eye.cpp").read_text(
        encoding="utf-8"
    )
    facade = (root / "native/filament/bridge/filament_bridge.cpp").read_text(
        encoding="utf-8"
    )
    public_header = (root / "native/filament/bridge/filament_bridge.h").read_text(
        encoding="utf-8"
    )

    assert "bridge_eye_create_controller_overlay_stereo_swapchain_with_depth" in eye_source
    assert "external_swapchain->depth =" in eye_source
    assert "external_swapchain->depth_format =" in eye_source
    assert "bridge->renderer->render(bridge->eyes[0].controller_view)" in eye_source
    assert "bridge->engine->flushAndWait()" in eye_source
    assert "0x02u | (1u << kScreenLayerBase)" in eye_source
    assert "eye.controller_view->setBlendMode(\n            filament::View::BlendMode::TRANSLUCENT)" in eye_source
    assert "eye.controller_view->setPostProcessingEnabled(false)" in eye_source
    assert "setChannelDepthClearEnabled(0, true)" in eye_source
    assert "filament_bridge_render_controller_composition_layer" in facade
    assert "filament_bridge_render_controller_composition_layer" in public_header
    assert (
        "filament_bridge_create_controller_overlay_stereo_swapchain_with_depth"
        in public_header
    )


def test_controller_and_unlit_laser_ignore_room_exposure() -> None:
    root = Path(__file__).resolve().parents[1]
    context_source = (root / "native/filament/bridge/bridge_context.cpp").read_text(
        encoding="utf-8"
    )
    material_source = (root / "native/filament/bridge/bridge_material.cpp").read_text(
        encoding="utf-8"
    )
    laser_source = (root / "native/filament/bridge/bridge_laser.cpp").read_text(
        encoding="utf-8"
    )

    assert ".exposure(0.0f)" in context_source
    assert "setColorGrading(eye.controller_color_grading)" in context_source
    exposure_body = material_source.split(
        "int bridge_material_set_scene_exposure", 1
    )[1].split("int bridge_material_set_skybox_brightness", 1)[0]
    assert "controller_view->setColorGrading" not in exposure_body
    assert "controller_guide_view->setColorGrading" not in exposure_body
    assert ".shading(filament::Shading::UNLIT)" in laser_source


def test_native_screen_has_opt_in_multiview_eye_diagnostic() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "native/filament/bridge/bridge_screen.cpp").exists()


@pytest.mark.parametrize("format_name", ("VK_FORMAT_R8G8B8A8_SRGB", "VK_FORMAT_R8G8B8A8_UNORM"))
def test_empty_eye_background_remains_exact_black_on_gpu(format_name: str) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("vulkan")
    from viewer.vulkan_context import (
        ImageState, VulkanCapabilityError, VulkanContext, VulkanUnavailableError,
    )
    from viewer.vulkan_resources import VulkanHostReadbackBuffer, VulkanTransientImage

    if not default_bridge_path().is_file():
        pytest.skip("The platform Filament bridge is not installed")
    try:
        context = VulkanContext.create()
    except (VulkanCapabilityError, VulkanUnavailableError) as exc:
        pytest.skip(f"A compatible Vulkan GPU is unavailable: {exc}")

    with context, ExitStack() as stack:
        vk = context.vk
        images = []
        for eye in range(2):
            image = VulkanTransientImage(context, 64, 64, format=getattr(vk, format_name), label=f"black-eye-{eye}")
            stack.callback(image.close)
            images.append(image)
        host = VulkanHostReadbackBuffer(context, 64, 64, label="black-eye-readback")
        stack.callback(host.close)
        bridge = FilamentVulkanBridge()
        stack.callback(bridge.close)
        bridge.create(
            instance=context.instance, physical_device=context.physical_device,
            device=context.device, queue_family_index=context.queue_family_index,
        )
        for eye, image in enumerate(images):
            bridge.create_eye_swapchain(eye, [image.image], format=image.format, width=64, height=64)
        for _frame in range(4):
            for eye, image in enumerate(images):
                bridge.set_active_eye(eye)
                bridge.set_camera_projection(90, 1)
                bridge.set_acquired_image(0)
                bridge.begin_frame()
                bridge.end_frame()
                context.register_image_state(image.image, ImageState(
                    layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    access_mask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    stage_mask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    queue_family_index=context.queue_family_index,
                ))
                timeline = context.copy_image_to_host_buffer(image.resource, host)
                context.wait_for_timeline(timeline)
                rgba = host.read_rgba()
                assert not np.any(rgba[:, :, :3]), "Empty OpenXR eye background must stay exactly black"
                assert np.all(rgba[:, :, 3] == 255), "Opaque background alpha must be preserved"
