#include "bridge_eye.h"
#include "bridge_internal.h"
#include "bridge_material.h"

#include <cstdlib>
#include <type_traits>

namespace {

template <typename T>
const void* semaphore_handle_as_void(T handle) {
    if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<const void*>(handle);
    } else {
        return reinterpret_cast<const void*>(static_cast<uintptr_t>(handle));
    }
}

void set_multiview_views(FilamentBridge* bridge, bool enabled) {
    if (!bridge) return;
    auto& eye = bridge->eyes[0];
    filament::StereoscopicOptions options{};
    options.enabled = enabled;
    eye.view->setStereoscopicOptions(options);
    eye.foreground_view->setStereoscopicOptions(options);
    eye.controller_view->setStereoscopicOptions(options);
    eye.controller_guide_view->setStereoscopicOptions(options);
    eye.view->setFrustumCullingEnabled(!enabled);
    eye.foreground_view->setFrustumCullingEnabled(!enabled);
    eye.controller_view->setFrustumCullingEnabled(!enabled);
    eye.controller_guide_view->setFrustumCullingEnabled(!enabled);
    // A translucent View forces Filament's final PostProcessManager::blit().
    // In multiview that path treats the whole array resource as subresource
    // layer zero, samples it with blitLow, and duplicates layer zero into both
    // views. Post-processing is disabled below, so render the layered
    // foreground directly and retain transparency through each material's own
    // blend state. Restore the legacy per-eye composition mode on fallback.
    const auto foreground_blend = enabled
            ? filament::View::BlendMode::OPAQUE
            : filament::View::BlendMode::TRANSLUCENT;
    eye.foreground_view->setBlendMode(foreground_blend);
    eye.controller_view->setBlendMode(foreground_blend);
    eye.controller_guide_view->setBlendMode(foreground_blend);
    eye.view->setPostProcessingEnabled(!enabled);
    eye.foreground_view->setPostProcessingEnabled(!enabled);
    eye.controller_view->setPostProcessingEnabled(!enabled);
    eye.controller_guide_view->setPostProcessingEnabled(!enabled);
}

void set_renderer_clear_alpha(FilamentBridge* bridge, double alpha) {
    if (!bridge || !bridge->renderer) return;
    filament::Renderer::ClearOptions clear_options;
    clear_options.clearColor = filament::math::double4{0.0, 0.0, 0.0, alpha};
    clear_options.clear = true;
    clear_options.discard = true;
    bridge->renderer->setClearOptions(clear_options);
}

filament::math::mat4 matrix_from_columns(const float* values) {
    return filament::math::mat4(filament::math::mat4f(
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7],
            values[8], values[9], values[10], values[11],
            values[12], values[13], values[14], values[15]));
}

} // namespace

void bridge_eye_activate(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->eyes.size()) return;
    auto& eye = bridge->eyes[eye_index];
    bridge->active_eye = eye_index;
    bridge->renderer = eye.renderer;
    bridge->view = eye.view;
    bridge->camera = eye.camera;
    bridge->color_grading = eye.color_grading;
    bridge->swapchain = eye.swapchain;
    bridge->external_swapchain = eye.external_swapchain;
    bridge->frame_active = eye.frame_active;
}

int bridge_eye_create_swapchain(
        FilamentBridge* bridge,
        const void* const* image_handles,
        uint32_t image_count,
        int32_t format,
        uint32_t width,
        uint32_t height) {
    return bridge_eye_create_target_swapchain(
            bridge, 0, image_handles, image_count, format, width, height);
}

int bridge_eye_create_target_swapchain(
        FilamentBridge* bridge, uint32_t eye_index,
        const void* const* image_handles, uint32_t image_count,
        int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format) {
    if (!bridge || !bridge->engine || !bridge->platform ||
            eye_index >= bridge->eyes.size()) return 0;
    auto& eye = bridge->eyes[eye_index];
    if (bridge->multiview_active) {
        set_multiview_views(bridge, false);
        bridge->multiview_active = false;
    }
    if (eye.swapchain) {
        bridge->engine->destroy(eye.swapchain);
        eye.swapchain = nullptr;
        eye.external_swapchain = nullptr;
    }
    bridge->depth_attachments[eye_index] = VK_NULL_HANDLE;
    bridge->depth_attachment_formats[eye_index] = VK_FORMAT_UNDEFINED;
    auto* external = bridge->platform->create_external_swapchain(
            image_handles, image_count, static_cast<VkFormat>(format), width, height);
    if (!external) {
        bridge_set_error(bridge, "Invalid OpenXR Vulkan swapchain image list");
        return 0;
    }
    uint64_t swapchain_flags = 0;
    if (static_cast<VkFormat>(format) == VK_FORMAT_R8G8B8A8_SRGB ||
            static_cast<VkFormat>(format) == VK_FORMAT_B8G8R8A8_SRGB) {
        swapchain_flags = filament::SwapChain::CONFIG_SRGB_COLORSPACE;
    }
    auto* external_swapchain =
            static_cast<OpenXrVulkanPlatform::ExternalSwapChain*>(external);
    external_swapchain->depth = depth_image_handle
            ? reinterpret_cast<VkImage>(const_cast<void*>(depth_image_handle))
            : VK_NULL_HANDLE;
    external_swapchain->depth_format = depth_image_handle
            ? static_cast<VkFormat>(depth_format)
            : VK_FORMAT_UNDEFINED;
    bridge->depth_attachments[eye_index] = external_swapchain->depth;
    bridge->depth_attachment_formats[eye_index] = external_swapchain->depth_format;
    eye.swapchain = bridge->engine->createSwapChain(external, swapchain_flags);
    if (!eye.swapchain) {
        bridge->platform->destroy(external);
        bridge_set_error(bridge, "Filament Vulkan SwapChain creation failed");
        return 0;
    }
    eye.external_swapchain =
            static_cast<OpenXrVulkanPlatform::ExternalSwapChain*>(external);
    std::fprintf(stderr,
            "[FilamentBridge] eye swapchain created eye=%u images=%u format=%d "
            "extent=%ux%u first_image=%p\n",
            eye_index, image_count, format, width, height,
            image_count ? reinterpret_cast<void*>(
                    reinterpret_cast<uintptr_t>(eye.external_swapchain->images[0]))
                         : nullptr);
    std::fflush(stderr);
    eye.camera->setProjection(
            45.0,
            static_cast<double>(width) / static_cast<double>(height),
            0.05,
            1000.0);
    eye.view->setViewport(filament::Viewport{0, 0, width, height});
    eye.foreground_view->setViewport(filament::Viewport{0, 0, width, height});
    eye.controller_view->setViewport(filament::Viewport{0, 0, width, height});
    eye.controller_guide_view->setViewport(filament::Viewport{0, 0, width, height});
    bridge_eye_activate(bridge, eye_index);
    set_renderer_clear_alpha(bridge, 1.0);
    return 1;
}

int bridge_eye_multiview_supported(const FilamentBridge* bridge) {
    return bridge && bridge->multiview_supported ? 1 : 0;
}

int bridge_eye_create_stereo_swapchain(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height) {
    return bridge_eye_create_stereo_swapchain_with_depth(
            bridge, image_handles, image_count, format, width, height,
            nullptr, 0);
}

int bridge_eye_create_stereo_swapchain_with_depth(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format) {
    if (!bridge || !bridge->engine || !bridge->platform ||
            !bridge->multiview_supported) return 0;
    auto& eye = bridge->eyes[0];
    if (eye.swapchain) {
        bridge->engine->destroy(eye.swapchain);
        eye.swapchain = nullptr;
        eye.external_swapchain = nullptr;
    }
    auto* external = bridge->platform->create_external_swapchain(
            image_handles, image_count, static_cast<VkFormat>(format),
            width, height, 2);
    if (!external) {
        bridge_set_error(bridge, "Invalid layered OpenXR Vulkan swapchain image list");
        return 0;
    }
    auto* external_swapchain =
            static_cast<OpenXrVulkanPlatform::ExternalSwapChain*>(external);
    external_swapchain->depth = depth_image_handle
            ? reinterpret_cast<VkImage>(const_cast<void*>(depth_image_handle))
            : VK_NULL_HANDLE;
    external_swapchain->depth_format = depth_image_handle
            ? static_cast<VkFormat>(depth_format)
            : VK_FORMAT_UNDEFINED;
    uint64_t swapchain_flags = 0;
    if (static_cast<VkFormat>(format) == VK_FORMAT_R8G8B8A8_SRGB ||
            static_cast<VkFormat>(format) == VK_FORMAT_B8G8R8A8_SRGB) {
        swapchain_flags = filament::SwapChain::CONFIG_SRGB_COLORSPACE;
    }
    eye.swapchain = bridge->engine->createSwapChain(external, swapchain_flags);
    if (!eye.swapchain) {
        bridge->platform->destroy(external);
        bridge_set_error(bridge, "Filament layered Vulkan SwapChain creation failed");
        return 0;
    }
    eye.external_swapchain = external_swapchain;
    eye.view->setViewport(filament::Viewport{0, 0, width, height});
    eye.foreground_view->setViewport(filament::Viewport{0, 0, width, height});
    eye.controller_view->setViewport(filament::Viewport{0, 0, width, height});
    eye.controller_guide_view->setViewport(filament::Viewport{0, 0, width, height});
    // Preserve the validated composition contract: room first, then one
    // foreground pass whose channel-2 depth is cleared before drawing Glow,
    // controllers, lasers and guides. The layered depth attachment supplied
    // above makes that same contract effective for both multiview layers.
    eye.foreground_view->setVisibleLayers(
            0xff, static_cast<uint8_t>(
                    0x01u | 0x02u | 0x04u | (1u << kScreenLayerBase)));
    set_multiview_views(bridge, true);
    bridge->multiview_active = true;
    bridge_eye_activate(bridge, 0);
    // The private HDR producer is composited over SBS/Glow later. Its empty
    // background must remain transparent while rendered geometry stays opaque.
    set_renderer_clear_alpha(bridge, 0.0);
    std::fprintf(stderr,
            "[FilamentBridge] stereo swapchain created images=%u format=%d "
            "extent=%ux%u layers=2 depth=%p depth_format=%d\n",
            image_count, format, width, height,
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(
                    external_swapchain->depth)),
            static_cast<int>(external_swapchain->depth_format));
    std::fflush(stderr);
    return 1;
}

int bridge_eye_create_controller_overlay_stereo_swapchain(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height) {
    return bridge_eye_create_controller_overlay_stereo_swapchain_with_depth(
            bridge, image_handles, image_count, format, width, height,
            nullptr, 0);
}

int bridge_eye_create_controller_overlay_stereo_swapchain_with_depth(
        FilamentBridge* bridge, const void* const* image_handles,
        uint32_t image_count, int32_t format, uint32_t width, uint32_t height,
        const void* depth_image_handle, int32_t depth_format) {
    if (!bridge || !bridge->engine || !bridge->platform ||
            !bridge->multiview_active || !image_handles || image_count == 0) {
        return 0;
    }
    if (bridge->controller_overlay_swapchain) {
        bridge->engine->destroy(bridge->controller_overlay_swapchain);
        bridge->controller_overlay_swapchain = nullptr;
        bridge->controller_overlay_external_swapchain = nullptr;
    }
    auto* external = bridge->platform->create_external_swapchain(
            image_handles, image_count, static_cast<VkFormat>(format),
            width, height, 2);
    if (!external) {
        bridge_set_error(
                bridge, "Invalid controller overlay OpenXR swapchain image list");
        return 0;
    }
    auto* external_swapchain =
            static_cast<OpenXrVulkanPlatform::ExternalSwapChain*>(external);
    external_swapchain->depth = depth_image_handle
            ? reinterpret_cast<VkImage>(const_cast<void*>(depth_image_handle))
            : VK_NULL_HANDLE;
    external_swapchain->depth_format = depth_image_handle
            ? static_cast<VkFormat>(depth_format)
            : VK_FORMAT_UNDEFINED;
    uint64_t swapchain_flags = 0;
    if (static_cast<VkFormat>(format) == VK_FORMAT_R8G8B8A8_SRGB ||
            static_cast<VkFormat>(format) == VK_FORMAT_B8G8R8A8_SRGB) {
        swapchain_flags = filament::SwapChain::CONFIG_SRGB_COLORSPACE;
    }
    bridge->controller_overlay_swapchain =
            bridge->engine->createSwapChain(external, swapchain_flags);
    if (!bridge->controller_overlay_swapchain) {
        bridge->platform->destroy(external);
        bridge_set_error(
                bridge, "Filament controller overlay SwapChain creation failed");
        return 0;
    }
    bridge->controller_overlay_external_swapchain =
            external_swapchain;
    auto& eye = bridge->eyes[0];
    eye.controller_view->setViewport(filament::Viewport{0, 0, width, height});
    // This swapchain is submitted as an alpha-blended OpenXR composition
    // layer. set_multiview_views(true) makes the shared views opaque for the
    // private HDR producer; restore transparent output for this dedicated
    // controller layer so its untouched pixels stay transparent.
    eye.controller_view->setBlendMode(
            filament::View::BlendMode::TRANSLUCENT);
    eye.controller_view->setPostProcessingEnabled(false);
    // The controller composition reuses the multiview depth attachment. Clear
    // both render channels before drawing so room/screen depth from the HDR
    // producer cannot hide the controller models or the B-button guide.
    eye.controller_view->setChannelDepthClearEnabled(0, true);
    eye.controller_view->setChannelDepthClearEnabled(2, true);
    // The main HDR producer retains only room/effect content. Controllers,
    // lasers and guides are emitted by the final transparent Projection layer.
    eye.foreground_view->setVisibleLayers(
            0xff, static_cast<uint8_t>(
                    0x02u | (1u << kScreenLayerBase)));
    std::fprintf(stderr,
            "[FilamentBridge] controller composition swapchain created "
            "images=%u format=%d extent=%ux%u layers=2 depth=%p "
            "depth_format=%d\n",
            image_count, format, width, height,
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(
                    external_swapchain->depth)),
            static_cast<int>(external_swapchain->depth_format));
    std::fflush(stderr);
    return 1;
}

int bridge_eye_set_controller_overlay_acquired_image(
        FilamentBridge* bridge, uint32_t image_index) {
    if (!bridge || !bridge->platform ||
            !bridge->controller_overlay_external_swapchain) return 0;
    return bridge->platform->set_pending_image(
            bridge->controller_overlay_external_swapchain, image_index) ? 1 : 0;
}

int bridge_eye_render_controller_composition_layer(FilamentBridge* bridge) {
    if (!bridge || !bridge->renderer || !bridge->engine ||
            !bridge->multiview_active || !bridge->controller_overlay_swapchain ||
            bridge->frame_active) return 0;
    set_renderer_clear_alpha(bridge, 0.0);
    const bool active = bridge->renderer->beginFrame(
            bridge->controller_overlay_swapchain);
    if (!active) {
        bridge_set_error(
                bridge, "Filament controller composition beginFrame failed");
        return 0;
    }
    bridge->renderer->render(bridge->eyes[0].controller_view);
    bridge->renderer->endFrame();
    // The OpenXR swapchain image is released immediately after this call.
    // Keep the proven synchronous contract and never route this through the
    // disabled dual-SwapChain deferred submission experiment.
    bridge->engine->flushAndWait();
    return 1;
}

int bridge_eye_set_active(FilamentBridge* bridge, uint32_t eye_index) {
    if (!bridge || eye_index >= bridge->eyes.size()) return 0;
    if (bridge->frame_active || bridge->eyes[eye_index].frame_active) return 0;
    bridge_eye_activate(bridge, eye_index);
    return 1;
}

int bridge_eye_set_acquired_image(FilamentBridge* bridge, uint32_t image_index) {
    if (!bridge || !bridge->swapchain || !bridge->platform) return 0;
    const int result = bridge->platform->set_pending_image(
            bridge->external_swapchain, image_index) ? 1 : 0;
    if (bridge->diagnostic_frame_count < 8) {
        const auto& images = bridge->external_swapchain->images;
        const void* image = image_index < images.size()
                ? reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(images[image_index]))
                : nullptr;
        std::fprintf(stderr,
                "[FilamentBridge] acquired eye=%u index=%u image=%p result=%d\n",
                bridge->active_eye, image_index, image, result);
        std::fflush(stderr);
    }
    return result;
}

int bridge_eye_set_camera_look_at(
        FilamentBridge* bridge,
        float eye_x, float eye_y, float eye_z,
        float center_x, float center_y, float center_z,
        float up_x, float up_y, float up_z) {
    if (!bridge || !bridge->camera) return 0;
    bridge->camera->lookAt(
            filament::math::float3{eye_x, eye_y, eye_z},
            filament::math::float3{center_x, center_y, center_z},
            filament::math::float3{up_x, up_y, up_z});
    if (bridge->active_eye == 0) {
        bridge_material_update_controller_lights(bridge, eye_x, eye_y, eye_z);
    }
    return 1;
}

int bridge_eye_set_camera_projection(
        FilamentBridge* bridge,
        double vertical_fov_degrees, double aspect,
        double near_plane, double far_plane) {
    if (!bridge || !bridge->camera || vertical_fov_degrees <= 0.0 ||
            aspect <= 0.0 || near_plane <= 0.0 || far_plane <= near_plane) {
        return 0;
    }
    bridge->camera->setProjection(
            vertical_fov_degrees, aspect, near_plane, far_plane);
    return 1;
}

int bridge_eye_set_camera_projection_frustum(
        FilamentBridge* bridge,
        double left, double right, double bottom, double top,
        double near_plane, double far_plane) {
    if (!bridge || !bridge->camera || right <= left || top <= bottom ||
            near_plane <= 0.0 || far_plane <= near_plane) {
        return 0;
    }
    bridge->camera->setProjection(
            filament::Camera::Projection::PERSPECTIVE,
            left, right, bottom, top, near_plane, far_plane);
    return 1;
}

int bridge_eye_set_stereo_camera(
        FilamentBridge* bridge, const float* eye_model_matrices32,
        const double* eye_frustums8, double near_plane, double far_plane) {
    if (!bridge || !bridge->multiview_active || !bridge->camera ||
            !eye_model_matrices32 || !eye_frustums8 || near_plane <= 0.0 ||
            far_plane <= near_plane) return 0;
    filament::math::mat4 eye_models[2] = {
            matrix_from_columns(eye_model_matrices32),
            matrix_from_columns(eye_model_matrices32 + 16)};
    filament::math::mat4 projections[2] = {
            filament::math::mat4::frustum(
                    eye_frustums8[0], eye_frustums8[1], eye_frustums8[2],
                    eye_frustums8[3], near_plane, far_plane),
            filament::math::mat4::frustum(
                    eye_frustums8[4], eye_frustums8[5], eye_frustums8[6],
                    eye_frustums8[7], near_plane, far_plane)};
    bridge->camera->setEyeModelMatrix(0, eye_models[0]);
    bridge->camera->setEyeModelMatrix(1, eye_models[1]);
    bridge->camera->setCustomEyeProjection(
            projections, 2, projections[0], near_plane, far_plane);
    static bool stereo_camera_trace_logged = false;
    if (!stereo_camera_trace_logged && std::getenv("D2S_FILAMENT_EYE_DIAGNOSTIC")) {
        stereo_camera_trace_logged = true;
        std::fprintf(stderr,
                "[D2S stereo trace] stereo camera "
                "eye0_t=(%.5f,%.5f,%.5f) eye1_t=(%.5f,%.5f,%.5f) "
                "eye0_lr=(%.5f,%.5f) eye1_lr=(%.5f,%.5f)\n",
                eye_model_matrices32[12], eye_model_matrices32[13],
                eye_model_matrices32[14], eye_model_matrices32[28],
                eye_model_matrices32[29], eye_model_matrices32[30],
                eye_frustums8[0], eye_frustums8[1],
                eye_frustums8[4], eye_frustums8[5]);
        std::fflush(stderr);
    }
    return 1;
}

namespace {

int bridge_eye_begin_frame_impl(
        FilamentBridge* bridge, bool render_controller_layers) {
    if (!bridge || !bridge->renderer || !bridge->swapchain || bridge->frame_active) {
        return 0;
    }
    bridge->frame_active = bridge->renderer->beginFrame(bridge->swapchain);
    bridge->eyes[bridge->active_eye].frame_active = bridge->frame_active;
    if (!bridge->frame_active) {
        bridge_set_error(bridge, "Filament Renderer::beginFrame failed");
        return 0;
    }
    if (bridge->diagnostic_frame_count < 8) {
        std::fprintf(stderr,
                "[FilamentBridge] begin eye=%u renderer=%p swapchain=%p active=%d\n",
                bridge->active_eye, static_cast<void*>(bridge->renderer),
                static_cast<void*>(bridge->swapchain), bridge->frame_active ? 1 : 0);
        std::fflush(stderr);
    }
    bridge->renderer->render(bridge->view);
    // The foreground View starts from fresh depth while preserving the room
    // color, matching the previously validated opaque-controller ordering.
    auto& eye = bridge->eyes[bridge->active_eye];
    bridge->renderer->render(eye.foreground_view);
    if (render_controller_layers && !bridge->multiview_active) {
        bridge->renderer->render(eye.controller_view);
    }
    return 1;
}

}  // namespace

int bridge_eye_begin_frame(FilamentBridge* bridge) {
    return bridge_eye_begin_frame_impl(bridge, true);
}

int bridge_eye_begin_background_frame(FilamentBridge* bridge) {
    return bridge_eye_begin_frame_impl(bridge, false);
}

int bridge_eye_render_controller_overlay(FilamentBridge* bridge) {
    if (!bridge || !bridge->renderer || !bridge->engine || !bridge->swapchain ||
            bridge->frame_active || bridge->multiview_active) return 0;
    auto& eye = bridge->eyes[bridge->active_eye];
    filament::Renderer::ClearOptions overlay_options;
    overlay_options.clear = false;
    overlay_options.discard = false;
    bridge->renderer->setClearOptions(overlay_options);
    bridge->frame_active = bridge->renderer->beginFrame(bridge->swapchain);
    eye.frame_active = bridge->frame_active;
    const bool rendered = bridge->frame_active;
    if (!bridge->frame_active) {
        bridge_set_error(bridge, "Filament controller overlay beginFrame failed");
    } else {
        bridge->renderer->render(eye.controller_view);
        bridge->renderer->endFrame();
        bridge->frame_active = false;
        eye.frame_active = false;
        bridge->engine->flushAndWait();
    }
    filament::Renderer::ClearOptions clear_options;
    // Vulkan interprets sRGB attachment clear values as linear RGB. Convert
    // the requested sRGB green (0, 153, 51) before clearing the eye swapchain.
    clear_options.clearColor = bridge->passthrough_backdrop
            ? filament::math::double4{
                    0.0, 0.31854677812509186, 0.033104766570885055, 1.0}
            : filament::math::double4{0.0, 0.0, 0.0, 1.0};
    clear_options.clear = true;
    clear_options.discard = true;
    bridge->renderer->setClearOptions(clear_options);
    return rendered ? 1 : 0;
}

namespace {

int bridge_eye_end_frame_impl(FilamentBridge* bridge, bool wait_for_idle) {
    if (!bridge || !bridge->renderer || !bridge->frame_active) return 0;
    bridge->renderer->endFrame();
    bridge->frame_active = false;
    bridge->eyes[bridge->active_eye].frame_active = false;
    if (!bridge->engine) return 0;
    if (wait_for_idle) {
        // Preserve the original ABI contract for callers that submit one eye
        // at a time or use an older Python presenter.
        bridge->engine->flushAndWait();
    } else {
        // Kick the shared Renderer's render thread now, but leave the owner
        // thread free to enqueue the second eye before the stereo pair waits.
        bridge->engine->flush();
    }
    if (bridge->diagnostic_frame_count < 8) {
        std::fprintf(stderr, "[FilamentBridge] end eye=%u deferred=%d\n",
                bridge->active_eye, wait_for_idle ? 0 : 1);
        std::fflush(stderr);
        if (bridge->multiview_active || bridge->active_eye == 1) {
            ++bridge->diagnostic_frame_count;
        }
    }
    return 1;
}

}  // namespace

int bridge_eye_end_frame(FilamentBridge* bridge) {
    return bridge_eye_end_frame_impl(bridge, true);
}

int bridge_eye_end_frame_deferred(FilamentBridge* bridge) {
    return bridge_eye_end_frame_impl(bridge, false);
}

int bridge_eye_finish_frame_batch(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine || bridge->frame_active) return 0;
    for (const auto& eye : bridge->eyes) {
        if (eye.frame_active) {
            bridge_set_error(bridge,
                    "Cannot finish Filament frame batch while an eye frame is active");
            return 0;
        }
    }
    bridge->engine->flushAndWait();
    return 1;
}

int bridge_eye_set_ready_semaphore(
        FilamentBridge* bridge, const void* semaphore) {
    if (!bridge || !bridge->platform || !bridge->swapchain || !semaphore) return 0;
    const auto ready = reinterpret_cast<VkSemaphore>(
            const_cast<void*>(semaphore));
    return bridge->platform->set_pending_ready_semaphore(bridge->external_swapchain, ready)
            ? 1 : 0;
}

int bridge_eye_get_finished_semaphore(
        FilamentBridge* bridge, const void** semaphore) {
    if (!bridge || !semaphore || !bridge->external_swapchain) return 0;
    const VkSemaphore finished = bridge->external_swapchain->last_finished_drawing;
    if (finished == VK_NULL_HANDLE) return 0;
    *semaphore = semaphore_handle_as_void(finished);
    return 1;
}
