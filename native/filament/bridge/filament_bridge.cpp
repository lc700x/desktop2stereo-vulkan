#include "filament_bridge.h"

#include "bridge_internal.h"

#include "bridge_context.h"
#include "bridge_controller.h"
#include "bridge_controller_guide.h"
#include "bridge_eye.h"
#include "bridge_laser.h"
#include "bridge_material.h"
#include "bridge_scene.h"
#include "bridge_text_overlay.h"
#include "preview_bridge.h"

FilamentBridge* filament_bridge_create_vulkan(
        const FilamentBridgeVulkanCreateInfo* info) {
    return bridge_context_create(info);
}

void filament_bridge_destroy(FilamentBridge* bridge) {
    bridge_context_destroy(bridge);
}

int filament_bridge_create_swapchain(
        FilamentBridge* bridge,
        const void* const* image_handles,
        uint32_t image_count,
        int32_t format,
        uint32_t width,
        uint32_t height) {
    return bridge_eye_create_swapchain(
            bridge, image_handles, image_count, format, width, height);
}

int filament_bridge_create_eye_swapchain(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height) {
    return bridge_eye_create_target_swapchain(
            bridge, eye_index, image_handles, image_count, format, width, height);
}

int filament_bridge_create_eye_swapchain_with_depth(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format) {
    return bridge_eye_create_target_swapchain(
            bridge, eye_index, image_handles, image_count, format, width, height,
            depth_image_handle, depth_format);
}

int filament_bridge_multiview_abi_available() { return 3; }
int filament_bridge_multiview_supported(const FilamentBridge* bridge) {
    return bridge_eye_multiview_supported(bridge); }
int filament_bridge_create_stereo_swapchain(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height) {
    return bridge_eye_create_stereo_swapchain(
            bridge, image_handles, image_count, format, width, height); }
int filament_bridge_create_stereo_swapchain_with_depth(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format) {
    return bridge_eye_create_stereo_swapchain_with_depth(
            bridge, image_handles, image_count, format, width, height,
            depth_image_handle, depth_format); }
int filament_bridge_create_controller_overlay_stereo_swapchain(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height) {
    return bridge_eye_create_controller_overlay_stereo_swapchain(
            bridge, image_handles, image_count, format, width, height); }
int filament_bridge_create_controller_overlay_stereo_swapchain_with_depth(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format) {
    return bridge_eye_create_controller_overlay_stereo_swapchain_with_depth(
            bridge, image_handles, image_count, format, width, height,
            depth_image_handle, depth_format); }
int filament_bridge_set_controller_overlay_acquired_image(
        FilamentBridge* bridge, uint32_t image_index) {
    return bridge_eye_set_controller_overlay_acquired_image(
            bridge, image_index); }
int filament_bridge_render_controller_composition_layer(FilamentBridge* bridge) {
    return bridge_eye_render_controller_composition_layer(bridge); }
int filament_bridge_set_active_eye(
        FilamentBridge* bridge, uint32_t eye_index) {
    return bridge_eye_set_active(bridge, eye_index);
}

int filament_bridge_set_acquired_image(
        FilamentBridge* bridge, uint32_t image_index) {
    return bridge_eye_set_acquired_image(bridge, image_index);
}

int filament_bridge_set_image_ready_semaphore(
        FilamentBridge* bridge, const void* semaphore) {
    return bridge_eye_set_ready_semaphore(bridge, semaphore);
}

int filament_bridge_set_camera_look_at(
        FilamentBridge* bridge,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    return bridge_eye_set_camera_look_at(
            bridge, eye_x, eye_y, eye_z,
            center_x, center_y, center_z, up_x, up_y, up_z);
}

int filament_bridge_set_camera_projection(
        FilamentBridge* bridge,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    return bridge_eye_set_camera_projection(
            bridge, vertical_fov_degrees, aspect, near_plane, far_plane);
}

int filament_bridge_set_camera_projection_frustum(
        FilamentBridge* bridge,
        double left, double right, double bottom, double top,
        double near_plane, double far_plane) {
    return bridge_eye_set_camera_projection_frustum(
            bridge, left, right, bottom, top, near_plane, far_plane);
}

int filament_bridge_set_stereo_camera(
        FilamentBridge* bridge, const float* eye_model_matrices32,
        const double* eye_frustums8, double near_plane, double far_plane) {
    return bridge_eye_set_stereo_camera(
            bridge, eye_model_matrices32, eye_frustums8, near_plane, far_plane); }

int filament_bridge_begin_frame(FilamentBridge* bridge) {
    return bridge_eye_begin_frame(bridge); }
int filament_bridge_begin_background_frame(FilamentBridge* bridge) {
    return bridge_eye_begin_background_frame(bridge); }
int filament_bridge_render_controller_overlay(FilamentBridge* bridge) {
    return bridge_eye_render_controller_overlay(bridge); }
int filament_bridge_render_controller_overlay_deferred(FilamentBridge* bridge) {
    return bridge_eye_render_controller_overlay_deferred(bridge); }
int filament_bridge_end_frame(FilamentBridge* bridge) {
    return bridge_eye_end_frame(bridge); }

int filament_bridge_end_frame_deferred(FilamentBridge* bridge) {
    return bridge_eye_end_frame_deferred(bridge); }
int filament_bridge_finish_frame_batch(FilamentBridge* bridge) {
    return bridge_eye_finish_frame_batch(bridge); }
int filament_bridge_wait_for_idle(FilamentBridge* bridge) {
    return bridge_context_wait_for_idle(bridge); }

int filament_bridge_load_glb(
        FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count) {
    return bridge_scene_load_glb(bridge, bytes, byte_count);
}

int filament_bridge_unload_glb(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return 0;
    bridge_scene_destroy(bridge);
    bridge->engine->flushAndWait();
    return 1;
}

int filament_bridge_load_controller(
        FilamentBridge* bridge, uint32_t hand,
        const uint8_t* bytes, uint32_t byte_count) {
    return bridge_controller_load(bridge, hand, bytes, byte_count);
}

int filament_bridge_set_controller_pose(
        FilamentBridge* bridge, uint32_t hand, const float* matrix16) {
    return bridge_controller_set_pose(bridge, hand, matrix16);
}

int filament_bridge_set_controller_inputs(
        FilamentBridge* bridge, uint32_t hand,
        float trigger, float grip,
        float joystick_x, float joystick_y,
        uint32_t button_mask) {
    return bridge_controller_set_inputs(
            bridge, hand, trigger, grip, joystick_x, joystick_y, button_mask);
}

int filament_bridge_set_controller_visible(
        FilamentBridge* bridge, uint32_t hand, int visible) {
    return bridge_controller_set_visible(bridge, hand, visible);
}

int filament_bridge_set_controller_material_override(
        FilamentBridge* bridge, float roughness_factor, float metallic_factor,
        float specular_red, float specular_green, float specular_blue) {
    return bridge_controller_set_material_override(
            bridge, roughness_factor, metallic_factor,
            specular_red, specular_green, specular_blue);
}

int filament_bridge_set_controller_laser(
        FilamentBridge* bridge, uint32_t hand,
        const float* matrix16, int visible) {
    return bridge_laser_set(bridge, hand, matrix16, visible);
}

int filament_bridge_set_controller_guide_texture(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height) {
    return bridge_controller_guide_set_texture(bridge, rgba, width, height);
}

int filament_bridge_set_controller_guide(
        FilamentBridge* bridge, const float* matrix16, int visible) {
    return bridge_controller_guide_set(bridge, matrix16, visible);
}

int filament_bridge_set_text_overlay_page_texture(
        FilamentBridge* bridge, uint32_t page,
        const uint8_t* rgba, uint32_t width, uint32_t height) {
    return bridge_text_overlay_set_page_texture(
            bridge, page, rgba, width, height);
}

int filament_bridge_set_text_overlay_page_vertices(
        FilamentBridge* bridge, uint32_t page,
        const float* vertices, uint32_t vertex_count,
        const uint16_t* indices, uint32_t index_count, int visible) {
    return bridge_text_overlay_set_page_vertices(
            bridge, page, vertices, vertex_count,
            indices, index_count, visible);
}

int filament_bridge_set_scene_exposure(
        FilamentBridge* bridge, float exposure_ev) {
    return bridge_material_set_scene_exposure(bridge, exposure_ev);
}

int filament_bridge_set_skybox_brightness(
        FilamentBridge* bridge, float brightness) {
    return bridge_material_set_skybox_brightness(bridge, brightness);
}

int filament_bridge_set_passthrough_backdrop(
        FilamentBridge* bridge, int enabled) {
    return bridge_material_set_passthrough_backdrop(bridge, enabled);
}

int filament_bridge_set_ambient_light(
        FilamentBridge* bridge, float red, float green, float blue) {
    return bridge_material_set_ambient_light(bridge, red, green, blue);
}

int filament_bridge_set_controller_ambient_light(
        FilamentBridge* bridge, float red, float green, float blue, int enabled) {
    return bridge_material_set_controller_ambient_light(
            bridge, red, green, blue, enabled);
}

int filament_bridge_set_lighting_config(
        FilamentBridge* bridge, const FilamentBridgeLightingConfig* config) {
    return bridge_material_set_lighting_config(bridge, config);
}

int filament_bridge_set_controller_screen_light(
        FilamentBridge* bridge, float red, float green, float blue,
        float intensity_lux, float direction_x, float direction_y,
        float direction_z, int cast_shadows, int enabled) {
    return bridge_material_set_controller_screen_light(
            bridge, red, green, blue, intensity_lux,
            direction_x, direction_y, direction_z, cast_shadows, enabled);
}

int filament_bridge_set_environment_screen_lights(
        FilamentBridge* bridge, const float* positions_xyz,
        const float* linear_rgb, const float* intensity_candela,
        uint32_t count, float falloff, int cast_shadows, int enabled) {
    return bridge_material_set_environment_screen_lights(
            bridge, positions_xyz, linear_rgb, intensity_candela,
            count, falloff, cast_shadows, enabled);
}

int filament_bridge_set_environment_screen_area_light(
        FilamentBridge* bridge, const float* position,
        const float* normal, const float* color,
        float half_width, float half_height, float intensity, int enabled) {
    return bridge_material_set_environment_screen_area_light(
            bridge, position, normal, color,
            half_width, half_height, intensity, enabled);
}

int filament_bridge_set_fill_light(
        FilamentBridge* bridge,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z) {
    return bridge_material_set_fill_light(
            bridge, red, green, blue, intensity,
            direction_x, direction_y, direction_z);
}

int filament_bridge_vulkan_external_image_abi_available(
        const FilamentBridge*) {
#if defined(D2S_FILAMENT_VULKAN_EXTERNAL_IMAGE)
    return 1;
#else
    // Filament v1.75's public Texture::Builder::import API does not accept
    // Vulkan VkImage handles. Keep the capability disabled until the Vulkan
    // backend is extended with a real external-image implementation.
    return 0;
#endif
}

int filament_bridge_depth_output_abi_available(const FilamentBridge*) {
    // The per-eye depth output contract has been validated on hardware.
    return 1;
}

int filament_bridge_get_depth_attachment(
        const FilamentBridge* bridge, uint32_t eye_index,
        const void** image_handle, int32_t* format) {
    if (!bridge || !image_handle || !format || eye_index >= bridge->eyes.size()) {
        return 0;
    }
    const auto* external = bridge->eyes[eye_index].external_swapchain;
    // Keep the active bridge alias as a compatibility fallback for older
    // per-eye activation paths. The eye-owned pointer remains authoritative.
    if (!external && bridge->active_eye == eye_index) {
        external = bridge->external_swapchain;
    }
    std::fprintf(
            stderr,
            "[FilamentBridge] depth query eye=%u external=%p depth=%p format=%d\n",
            eye_index,
            static_cast<const void*>(external),
            reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(
                    bridge->depth_attachments[eye_index])),
            static_cast<int>(bridge->depth_attachment_formats[eye_index]));
    std::fflush(stderr);
    const VkImage depth = bridge->depth_attachments[eye_index] != VK_NULL_HANDLE
            ? bridge->depth_attachments[eye_index]
            : (external ? external->depth : VK_NULL_HANDLE);
    const VkFormat depth_format =
            bridge->depth_attachment_formats[eye_index] != VK_FORMAT_UNDEFINED
            ? bridge->depth_attachment_formats[eye_index]
            : (external ? external->depth_format : VK_FORMAT_UNDEFINED);
    if (depth == VK_NULL_HANDLE || depth_format == VK_FORMAT_UNDEFINED) {
        return 0;
    }
    *image_handle = reinterpret_cast<const void*>(
            reinterpret_cast<uintptr_t>(depth));
    *format = static_cast<int32_t>(depth_format);
    return 1;
}

int filament_bridge_get_finished_drawing_semaphore(
        FilamentBridge* bridge, const void** semaphore) {
    return bridge_eye_get_finished_semaphore(bridge, semaphore);
}

int filament_bridge_apply_animations(
        FilamentBridge* bridge, double time_seconds) {
    return bridge_scene_apply_animations(bridge, time_seconds);
}

uint32_t filament_bridge_animation_count(const FilamentBridge* bridge) {
    return bridge_scene_animation_count(bridge);
}

float filament_bridge_animation_duration(
        const FilamentBridge* bridge, uint32_t animation_index) {
    return bridge_scene_animation_duration(bridge, animation_index);
}

const char* filament_bridge_last_error(const FilamentBridge* bridge) {
    return bridge_context_last_error(bridge);
}

FilamentPreview* filament_preview_create(
        void* native_window, uint32_t width, uint32_t height) {
    return preview_bridge_create(native_window, width, height);
}

void filament_preview_destroy(FilamentPreview* preview) {
    preview_bridge_destroy(preview);
}

int filament_preview_load_glb(
        FilamentPreview* preview, const uint8_t* bytes, uint32_t byte_count) {
    return preview_bridge_load_glb(preview, bytes, byte_count);
}

int filament_preview_apply_animations(
        FilamentPreview* preview, double time_seconds) {
    return preview_bridge_apply_animations(preview, time_seconds);
}

int filament_preview_set_camera(
        FilamentPreview* preview,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    return preview_bridge_set_camera(
            preview, eye_x, eye_y, eye_z,
            center_x, center_y, center_z, up_x, up_y, up_z);
}

int filament_preview_set_projection(
        FilamentPreview* preview,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    return preview_bridge_set_projection(
            preview, vertical_fov_degrees, aspect, near_plane, far_plane);
}

int filament_preview_set_viewport(
        FilamentPreview* preview, uint32_t width, uint32_t height) {
    return preview_bridge_set_viewport(preview, width, height);
}

int filament_preview_set_scene_exposure(
        FilamentPreview* preview, float exposure_ev) {
    return preview_material_set_scene_exposure(preview, exposure_ev);
}

int filament_preview_set_ambient_light(
        FilamentPreview* preview, float red, float green, float blue) {
    return preview_material_set_ambient_light(preview, red, green, blue);
}

int filament_preview_set_ambient_light_with_intensity(
        FilamentPreview* preview, float red, float green, float blue,
        float intensity_lux) {
    return preview_material_set_ambient_light_with_intensity(
            preview, red, green, blue, intensity_lux);
}

int filament_preview_set_fill_light(
        FilamentPreview* preview,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z) {
    return preview_material_set_fill_light(
            preview, red, green, blue, intensity,
            direction_x, direction_y, direction_z);
}

int filament_preview_set_skybox_brightness(
        FilamentPreview* preview, float brightness) {
    return preview_material_set_skybox_brightness(preview, brightness);
}

int filament_preview_set_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees,
        float rotation_z_degrees) {
    return preview_bridge_set_screen(
            preview, position_x, position_y, position_z, width, height,
            rotation_x_degrees, rotation_y_degrees, rotation_z_degrees);
}

int filament_preview_render(FilamentPreview* preview) {
    return preview_bridge_render(preview);
}

const char* filament_preview_last_error(const FilamentPreview* preview) {
    return preview_bridge_last_error(preview);
}
