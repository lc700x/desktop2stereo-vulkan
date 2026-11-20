#pragma once

#include <stdint.h>

#if defined(_WIN32)
#if defined(FILAMENT_BRIDGE_BUILD)
#define FILAMENT_BRIDGE_API __declspec(dllexport)
#else
#define FILAMENT_BRIDGE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define FILAMENT_BRIDGE_API __attribute__((visibility("default")))
#else
#define FILAMENT_BRIDGE_API
#endif

extern "C" {

struct FilamentBridge;
struct FilamentPreview;

// Handles are borrowed from the Python-owned OpenXR Vulkan session.
struct FilamentBridgeVulkanCreateInfo {
    void* instance;
    void* physical_device;
    void* device;
    uint32_t graphics_queue_family_index;
    uint32_t graphics_queue_index;
};

// All appearance values are supplied by the host profile. The bridge only
// validates and applies physical light parameters; it owns no visual defaults.
struct FilamentBridgeLightingConfig {
    uint32_t struct_size;
    float environment_ambient_color[3];
    float environment_ambient_intensity_lux;
    float controller_ambient_color[3];
    float controller_ambient_intensity_lux;
    int32_t controller_ambient_enabled;
    float head_light_color[3];
    float head_light_intensity_candela;
    float head_light_offset[3];
    float head_light_falloff;
    int32_t head_light_cast_shadows;
    float top_light_color[3];
    float top_light_intensity_candela;
    float top_light_offset[3];
    float top_light_falloff;
    int32_t top_light_cast_shadows;
};

FILAMENT_BRIDGE_API FilamentBridge* filament_bridge_create_vulkan(
        const FilamentBridgeVulkanCreateInfo* info);
FILAMENT_BRIDGE_API void filament_bridge_destroy(FilamentBridge* bridge);

// image_handles points to borrowed VkImage handles owned by an OpenXR swapchain.
FILAMENT_BRIDGE_API int filament_bridge_create_swapchain(
        FilamentBridge* bridge,
        const void* const* image_handles,
        uint32_t image_count,
        int32_t format,
        uint32_t width,
        uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_create_eye_swapchain(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_create_eye_swapchain_with_depth(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format);
FILAMENT_BRIDGE_API int filament_bridge_multiview_abi_available();
FILAMENT_BRIDGE_API int filament_bridge_multiview_supported(
        const FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_create_stereo_swapchain(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_create_stereo_swapchain_with_depth(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format);
// Bind a second array_size=2 OpenXR swapchain used only for the final
// controller/laser/guide Projection composition layer.
FILAMENT_BRIDGE_API int filament_bridge_create_controller_overlay_stereo_swapchain(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_create_controller_overlay_stereo_swapchain_with_depth(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_overlay_acquired_image(
        FilamentBridge* bridge, uint32_t image_index);
FILAMENT_BRIDGE_API int filament_bridge_render_controller_composition_layer(
        FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_set_active_eye(
        FilamentBridge* bridge, uint32_t eye_index);
FILAMENT_BRIDGE_API int filament_bridge_set_acquired_image(
        FilamentBridge* bridge, uint32_t image_index);
// Supply the binary semaphore that makes the selected borrowed image writable.
FILAMENT_BRIDGE_API int filament_bridge_set_image_ready_semaphore(
        FilamentBridge* bridge, const void* semaphore);
FILAMENT_BRIDGE_API int filament_bridge_set_camera_look_at(
        FilamentBridge* bridge,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z);
FILAMENT_BRIDGE_API int filament_bridge_set_camera_projection(
        FilamentBridge* bridge,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane);
FILAMENT_BRIDGE_API int filament_bridge_set_camera_projection_frustum(
        FilamentBridge* bridge,
        double left, double right, double bottom, double top,
        double near_plane, double far_plane);
FILAMENT_BRIDGE_API int filament_bridge_set_stereo_camera(
        FilamentBridge* bridge, const float* eye_model_matrices32,
        const double* eye_frustums8, double near_plane, double far_plane);
FILAMENT_BRIDGE_API int filament_bridge_begin_frame(FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_begin_background_frame(
        FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_render_controller_overlay(
        FilamentBridge* bridge);
// Queue a controller eye without waiting; finish_frame_batch synchronizes the
// complete stereo pair before the OpenXR swapchain images are released.
FILAMENT_BRIDGE_API int filament_bridge_render_controller_overlay_deferred(
        FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_end_frame(FilamentBridge* bridge);
// Queue one eye without blocking, then wait once after both eyes are queued.
FILAMENT_BRIDGE_API int filament_bridge_end_frame_deferred(FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_finish_frame_batch(FilamentBridge* bridge);
// Submit queued work without blocking; wait once after the complete XR frame.
FILAMENT_BRIDGE_API int filament_bridge_wait_for_idle(FilamentBridge* bridge);

FILAMENT_BRIDGE_API int filament_bridge_load_glb(
        FilamentBridge* bridge, const uint8_t* bytes, uint32_t byte_count);
FILAMENT_BRIDGE_API int filament_bridge_unload_glb(FilamentBridge* bridge);

// Controller assets are loaded into the same Filament scene as the environment.
FILAMENT_BRIDGE_API int filament_bridge_load_controller(
        FilamentBridge* bridge, uint32_t hand,
        const uint8_t* bytes, uint32_t byte_count);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_pose(
        FilamentBridge* bridge, uint32_t hand,
        const float* matrix16);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_inputs(
        FilamentBridge* bridge, uint32_t hand,
        float trigger, float grip,
        float joystick_x, float joystick_y,
        uint32_t button_mask);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_visible(
        FilamentBridge* bridge, uint32_t hand, int visible);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_material_override(
        FilamentBridge* bridge, float roughness_factor, float metallic_factor,
        float specular_red, float specular_green, float specular_blue);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_laser(
        FilamentBridge* bridge, uint32_t hand,
        const float* matrix16, int visible);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_guide_texture(
        FilamentBridge* bridge, const uint8_t* rgba,
        uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_guide(
        FilamentBridge* bridge, const float* matrix16, int visible);

// MSDF text is submitted as world-space glyph quads. The atlas page texture
// is uploaded once, while vertices are updated on the Presenter thread.
FILAMENT_BRIDGE_API int filament_bridge_set_text_overlay_page_texture(
        FilamentBridge* bridge, uint32_t page,
        const uint8_t* rgba, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_bridge_set_text_overlay_page_vertices(
        FilamentBridge* bridge, uint32_t page,
        const float* vertices, uint32_t vertex_count,
        const uint16_t* indices, uint32_t index_count, int visible);

FILAMENT_BRIDGE_API int filament_bridge_set_scene_exposure(
        FilamentBridge* bridge, float exposure_ev);
FILAMENT_BRIDGE_API int filament_bridge_set_skybox_brightness(
        FilamentBridge* bridge, float brightness);
FILAMENT_BRIDGE_API int filament_bridge_set_passthrough_backdrop(
        FilamentBridge* bridge, int enabled);
FILAMENT_BRIDGE_API int filament_bridge_set_ambient_light(
        FilamentBridge* bridge, float red, float green, float blue);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_ambient_light(
        FilamentBridge* bridge, float red, float green, float blue, int enabled);
FILAMENT_BRIDGE_API int filament_bridge_set_lighting_config(
        FilamentBridge* bridge, const FilamentBridgeLightingConfig* config);
FILAMENT_BRIDGE_API int filament_bridge_set_controller_screen_light(
        FilamentBridge* bridge, float red, float green, float blue,
        float intensity_lux, float direction_x, float direction_y,
        float direction_z, int cast_shadows, int enabled);
// Apply up to 24 screen-edge samples as local point lights on environment
// channel 0. Positions and linear RGB colors contain count packed vec3 values.
FILAMENT_BRIDGE_API int filament_bridge_set_environment_screen_lights(
        FilamentBridge* bridge, const float* positions_xyz,
        const float* linear_rgb, const float* intensity_candela,
        uint32_t count, float falloff, int cast_shadows, int enabled);
// Apply the virtual screen as one continuous rectangular area light through
// material globals. This is independent from Filament punctual lights.
FILAMENT_BRIDGE_API int filament_bridge_set_environment_screen_area_light(
        FilamentBridge* bridge, const float* position,
        const float* normal, const float* color,
        float half_width, float half_height, float intensity, int enabled);
FILAMENT_BRIDGE_API int filament_bridge_set_fill_light(
        FilamentBridge* bridge,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z);
// Reports whether this Bridge is backed by a Filament Vulkan backend that
// supports sampling an application-owned VkImage as a material texture.
FILAMENT_BRIDGE_API int filament_bridge_vulkan_external_image_abi_available(
        const FilamentBridge* bridge);
// Reports whether the bridge exports producer-owned per-eye depth images.
// Color-only external swapchains must return 0.
FILAMENT_BRIDGE_API int filament_bridge_depth_output_abi_available(
        const FilamentBridge* bridge);
FILAMENT_BRIDGE_API int filament_bridge_get_depth_attachment(
        const FilamentBridge* bridge, uint32_t eye_index,
        const void** image_handle, int32_t* format);
// Return the borrowed Filament render-finished semaphore for the active eye.
// The caller must consume it before the next frame reuses the swapchain slot.
FILAMENT_BRIDGE_API int filament_bridge_get_finished_drawing_semaphore(
        FilamentBridge* bridge, const void** semaphore);
FILAMENT_BRIDGE_API int filament_bridge_apply_animations(
        FilamentBridge* bridge, double time_seconds);
FILAMENT_BRIDGE_API uint32_t filament_bridge_animation_count(
        const FilamentBridge* bridge);
FILAMENT_BRIDGE_API float filament_bridge_animation_duration(
        const FilamentBridge* bridge, uint32_t animation_index);
FILAMENT_BRIDGE_API const char* filament_bridge_last_error(
        const FilamentBridge* bridge);

FILAMENT_BRIDGE_API FilamentPreview* filament_preview_create(
        void* native_window, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API void filament_preview_destroy(FilamentPreview* preview);
FILAMENT_BRIDGE_API int filament_preview_load_glb(
        FilamentPreview* preview, const uint8_t* bytes, uint32_t byte_count);
FILAMENT_BRIDGE_API int filament_preview_apply_animations(
        FilamentPreview* preview, double time_seconds);
FILAMENT_BRIDGE_API int filament_preview_set_camera(
        FilamentPreview* preview,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z);
FILAMENT_BRIDGE_API int filament_preview_set_projection(
        FilamentPreview* preview,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane);
FILAMENT_BRIDGE_API int filament_preview_set_viewport(
        FilamentPreview* preview, uint32_t width, uint32_t height);
FILAMENT_BRIDGE_API int filament_preview_set_scene_exposure(
        FilamentPreview* preview, float exposure_ev);
FILAMENT_BRIDGE_API int filament_preview_set_ambient_light(
        FilamentPreview* preview, float red, float green, float blue);
FILAMENT_BRIDGE_API int filament_preview_set_ambient_light_with_intensity(
        FilamentPreview* preview, float red, float green, float blue,
        float intensity_lux);
FILAMENT_BRIDGE_API int filament_preview_set_fill_light(
        FilamentPreview* preview,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z);
FILAMENT_BRIDGE_API int filament_preview_set_skybox_brightness(
        FilamentPreview* preview, float brightness);
FILAMENT_BRIDGE_API int filament_preview_set_screen(
        FilamentPreview* preview,
        float position_x, float position_y, float position_z,
        float width, float height,
        float rotation_x_degrees, float rotation_y_degrees, float rotation_z_degrees);
FILAMENT_BRIDGE_API int filament_preview_render(FilamentPreview* preview);
FILAMENT_BRIDGE_API const char* filament_preview_last_error(
        const FilamentPreview* preview);

}
