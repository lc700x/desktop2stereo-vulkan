#include "bridge_context.h"
#include "bridge_internal.h"
#include "bridge_controller.h"
#include "bridge_controller_guide.h"
#include "bridge_eye.h"
#include "bridge_laser.h"
#include "bridge_material.h"
#include "bridge_material_provider.h"
#include "bridge_scene.h"
#include "bridge_text_overlay.h"

void bridge_set_error(FilamentBridge* bridge, const char* message) {
    if (bridge) {
        bridge->last_error = message;
    }
}

void bridge_set_renderable_visible(
        FilamentBridge* bridge, utils::Entity entity, bool visible) {
    bridge_set_renderable_layer(bridge, entity, 0, visible);
}

void bridge_set_renderable_layer(
        FilamentBridge* bridge, utils::Entity entity,
        uint8_t layer, bool visible) {
    if (!bridge || !bridge->engine || entity.isNull()) return;
    auto& renderables = bridge->engine->getRenderableManager();
    const auto instance = renderables.getInstance(entity);
    if (!instance.isValid()) return;
    const uint8_t layer_mask = static_cast<uint8_t>(1u << layer);
    renderables.setLayerMask(instance, 0xff, visible ? layer_mask : 0x00);
}

FilamentBridge* bridge_context_create(
        const FilamentBridgeVulkanCreateInfo* info) {
    auto bridge = std::make_unique<FilamentBridge>();
    if (!info || !info->instance || !info->physical_device || !info->device) {
        bridge_set_error(bridge.get(), "Vulkan create info contains a null handle");
        return bridge.release();
    }

    bridge->shared_context.instance = reinterpret_cast<VkInstance>(info->instance);
    bridge->shared_context.physicalDevice = reinterpret_cast<VkPhysicalDevice>(
            info->physical_device);
    bridge->shared_context.logicalDevice = reinterpret_cast<VkDevice>(info->device);
    bridge->shared_context.graphicsQueueFamilyIndex =
            info->graphics_queue_family_index;
    bridge->shared_context.graphicsQueueIndex = info->graphics_queue_index;
    bridge->platform = new OpenXrVulkanPlatform();
    filament::Engine::Config engine_config{};
    engine_config.stereoscopicType = filament::Engine::StereoscopicType::MULTIVIEW;
    engine_config.stereoscopicEyeCount = 2;
    bridge->engine = filament::Engine::Builder()
            .backend(filament::Engine::Backend::VULKAN)
            .platform(bridge->platform)
            .sharedContext(&bridge->shared_context)
            .config(&engine_config)
            .build();
    if (!bridge->engine) {
        bridge_set_error(bridge.get(), "Filament Vulkan Engine creation failed");
        delete bridge->platform;
        bridge->platform = nullptr;
        return bridge.release();
    }
    bridge->multiview_supported = bridge->engine->isStereoSupported(
            filament::Engine::StereoscopicType::MULTIVIEW);
    bridge->scene = bridge->engine->createScene();
    bridge->foreground_scene = bridge->engine->createScene();
    bridge->materials = bridge_create_material_provider(bridge->engine);
    bridge->texture_provider = filament::gltfio::createStbProvider(bridge->engine);
    if (!bridge->scene || !bridge->foreground_scene || !bridge->materials ||
            !bridge->texture_provider) {
        bridge_set_error(bridge.get(), "Filament Vulkan resource creation failed");
        return bridge.release();
    }
    // Filament's Vulkan backend is not safe for two Renderer frames that are
    // simultaneously in-flight on the same Engine. Both OpenXR eyes share one
    // Renderer and only differ by View/Camera/SwapChain; deferred submission
    // remains possible because endFrame() is followed by a single flushAndWait.
    auto* shared_renderer = bridge->engine->createRenderer();
    if (!shared_renderer) {
        bridge_set_error(bridge.get(), "Filament Vulkan renderer creation failed");
        return bridge.release();
    }
    for (uint32_t eye_index = 0; eye_index < bridge->eyes.size(); ++eye_index) {
        auto& eye = bridge->eyes[eye_index];
        eye.renderer = shared_renderer;
        eye.view = bridge->engine->createView();
        eye.foreground_view = bridge->engine->createView();
        eye.controller_view = bridge->engine->createView();
        eye.controller_guide_view = bridge->engine->createView();
        eye.camera = bridge->engine->createCamera(
                utils::EntityManager::get().create());
        if (!eye.view || !eye.foreground_view ||
                !eye.controller_view || !eye.controller_guide_view || !eye.camera) {
            bridge_set_error(bridge.get(), "Filament Vulkan eye resource creation failed");
            return bridge.release();
        }
        // Output stays linear until the sRGB target store. Filament's 8-bit
        // dithering assumes an already encoded output; here it lifts black
        // pixels and adds noise even to untouched transparent overlays.
        eye.view->setDithering(filament::View::Dithering::NONE);
        eye.foreground_view->setDithering(filament::View::Dithering::NONE);
        eye.controller_view->setDithering(filament::View::Dithering::NONE);
        eye.controller_guide_view->setDithering(filament::View::Dithering::NONE);
        // Renderer::ClearOptions defaults to clear=false. OpenXR supplies an
        // external image ring whose previous color content is undefined for
        // the next frame, so every eye renderer must clear it explicitly.
        filament::Renderer::ClearOptions clear_options;
        clear_options.clearColor = filament::math::double4{0.0, 0.0, 0.0, 1.0};
        clear_options.clear = true;
        clear_options.discard = true;
        eye.renderer->setClearOptions(clear_options);
        eye.camera->lookAt(
                filament::math::float3{0.0f, 0.0f, 3.0f},
                filament::math::float3{0.0f, 0.0f, 0.0f},
                filament::math::float3{0.0f, 1.0f, 0.0f});
        eye.view->setScene(bridge->scene);
        eye.view->setCamera(eye.camera);
        eye.view->setVisibleLayers(0xff, 0x03);
        // The legacy OpenGL projection path has no post-process FXAA. Keep
        // text and fine screen details from being softened by Filament's
        // default FXAA pass.
        eye.view->setAntiAliasing(filament::AntiAliasing::NONE);
        // There is only one View per eye. The old false setting was required
        // to retain depth between the former scene and laser Views; retaining
        // it here leaks external OpenXR depth across frames and creates trails.
        // RenderableManager::Builder defaults imported GLB renderables to
        // channel 2, so depth clearing must target that channel.
        eye.view->setChannelDepthClearEnabled(2, true);
        eye.foreground_view->setScene(bridge->foreground_scene);
        eye.foreground_view->setCamera(eye.camera);
        // Layer 1 contains transparent Glow; each eye also sees only its own
        // screen layer. Render them before layer 0 controllers.
        eye.foreground_view->setVisibleLayers(
                0xff, static_cast<uint8_t>(
                        0x02u | (1u << (kScreenLayerBase + eye_index))));
        eye.foreground_view->setAntiAliasing(filament::AntiAliasing::NONE);
        // PBR foreground assets must use the same exposure and output color
        // transform as the room. A translucent View keeps untouched pixels
        // transparent so the post-processed foreground composites over the
        // room pass instead of replacing it.
        eye.foreground_view->setBlendMode(filament::View::BlendMode::TRANSLUCENT);
        eye.foreground_view->setPostProcessingEnabled(true);
        // Start the foreground pass with a fresh depth buffer. The room pass
        // owns its depth, but reusing it here causes room geometry to clip
        // controller surfaces and makes opaque controllers look transparent.
        // The renderer clear options keep the color buffer intact. Imported
        // foreground renderables use Filament's default render channel 2.
        eye.foreground_view->setChannelDepthClearEnabled(2, true);
        eye.controller_view->setScene(bridge->foreground_scene);
        eye.controller_view->setCamera(eye.camera);
        // Controllers/lasers use layer 0 and the guide uses layer 2. The
        // guide material disables depth culling, so both can share one View
        // without allowing the controller to cover the guide. This avoids a
        // second full-resolution post-process View for every eye.
        eye.controller_view->setVisibleLayers(0xff, 0x05);
        eye.controller_view->setAntiAliasing(filament::AntiAliasing::NONE);
        eye.controller_view->setBlendMode(filament::View::BlendMode::TRANSLUCENT);
        eye.controller_view->setPostProcessingEnabled(true);
        // Controllers and laser render after the screen and Glow, starting
        // from fresh depth so those earlier passes cannot clip the hands.
        eye.controller_view->setChannelDepthClearEnabled(2, true);
        eye.controller_guide_view->setScene(bridge->foreground_scene);
        eye.controller_guide_view->setCamera(eye.camera);
        eye.controller_guide_view->setVisibleLayers(0xff, 0x04);
        eye.controller_guide_view->setAntiAliasing(filament::AntiAliasing::NONE);
        eye.controller_guide_view->setBlendMode(
                filament::View::BlendMode::TRANSLUCENT);
        eye.controller_guide_view->setPostProcessingEnabled(true);
        // Retain the dedicated guide View for ABI/resource compatibility;
        // normal rendering includes layer 2 in controller_view above.
        eye.controller_guide_view->setChannelDepthClearEnabled(2, true);
    }
    bridge_eye_activate(bridge.get(), 0);
    for (auto& eye : bridge->eyes) {
        bridge->view = eye.view;
        bridge->camera = eye.camera;
        bridge->color_grading = nullptr;
        if (!bridge_material_configure_color_pipeline(bridge.get())) {
            bridge_set_error(bridge.get(), "Filament Vulkan color pipeline creation failed");
            return bridge.release();
        }
        eye.color_grading = bridge->color_grading;
        eye.foreground_view->setColorGrading(eye.color_grading);
        eye.controller_color_grading = filament::ColorGrading::Builder()
                .toneMapping(filament::ColorGrading::ToneMapping::LINEAR)
                .exposure(0.0f)
                .outputColorSpace(
                        filament::color::Rec709 - filament::color::Linear -
                        filament::color::D65)
                .build(*bridge->engine);
        if (!eye.controller_color_grading) {
            bridge_set_error(
                    bridge.get(),
                    "Filament neutral controller color pipeline creation failed");
            return bridge.release();
        }
        eye.controller_view->setColorGrading(eye.controller_color_grading);
        eye.controller_guide_view->setColorGrading(
                eye.controller_color_grading);
    }
    bridge_eye_activate(bridge.get(), 0);
    filament::gltfio::AssetConfiguration config{bridge->engine, bridge->materials};
    bridge->asset_loader = filament::gltfio::AssetLoader::create(config);
    if (!bridge->asset_loader) {
        bridge_set_error(bridge.get(), "Filament AssetLoader creation failed");
    } else if (!bridge_laser_create(bridge.get())) {
        return bridge.release();
    }
    return bridge.release();
}

void bridge_context_destroy(FilamentBridge* bridge) {
    if (!bridge) return;
    bridge_scene_destroy(bridge);
    for (auto& controller : bridge->controllers) {
        bridge_controller_destroy(bridge, controller);
    }
    bridge_controller_eye_diagnostic_destroy(bridge);
    bridge_laser_destroy(bridge);
    bridge_controller_guide_destroy(bridge);
    bridge_text_overlay_destroy(bridge);
    if (bridge->renderer && bridge->engine) {
        bridge->engine->destroy(bridge->renderer);
        bridge->renderer = nullptr;
    }
    if (bridge->controller_overlay_swapchain && bridge->engine) {
        bridge->engine->destroy(bridge->controller_overlay_swapchain);
        bridge->controller_overlay_swapchain = nullptr;
        bridge->controller_overlay_external_swapchain = nullptr;
    }
    for (auto& eye : bridge->eyes) {
        eye.renderer = nullptr;
        if (eye.swapchain && bridge->engine) {
            bridge->engine->destroy(eye.swapchain);
        }
        if (eye.color_grading && bridge->engine) {
            if (eye.view) eye.view->setColorGrading(nullptr);
            if (eye.foreground_view) eye.foreground_view->setColorGrading(nullptr);
            bridge->engine->destroy(eye.color_grading);
        }
        if (eye.controller_color_grading && bridge->engine) {
            if (eye.controller_view) eye.controller_view->setColorGrading(nullptr);
            if (eye.controller_guide_view) {
                eye.controller_guide_view->setColorGrading(nullptr);
            }
            bridge->engine->destroy(eye.controller_color_grading);
        }
        if (eye.view && bridge->engine) {
            bridge->engine->destroy(eye.view);
        }
        if (eye.foreground_view && bridge->engine) {
            bridge->engine->destroy(eye.foreground_view);
        }
        if (eye.controller_view && bridge->engine) {
            bridge->engine->destroy(eye.controller_view);
        }
        if (eye.controller_guide_view && bridge->engine) {
            bridge->engine->destroy(eye.controller_guide_view);
        }
        if (eye.camera && bridge->engine) {
            bridge->engine->destroy(eye.camera->getEntity());
        }
    }
    if (!bridge->fill_light.isNull() && bridge->engine) {
        bridge->engine->destroy(bridge->fill_light);
    }
    if (!bridge->controller_top_light.isNull() && bridge->engine) {
        bridge->engine->destroy(bridge->controller_top_light);
    }
    if (!bridge->controller_screen_light.isNull() && bridge->engine) {
        bridge->engine->destroy(bridge->controller_screen_light);
    }
    for (auto& light : bridge->environment_screen_lights) {
        if (!light.isNull() && bridge->engine) {
            bridge->engine->destroy(light);
            light = {};
        }
    }
    if (bridge->indirect_light && bridge->engine) {
        if (bridge->scene) bridge->scene->setIndirectLight(nullptr);
        bridge->engine->destroy(bridge->indirect_light);
        bridge->indirect_light = nullptr;
    }
    if (bridge->controller_indirect_light && bridge->engine) {
        if (bridge->foreground_scene) {
            bridge->foreground_scene->setIndirectLight(nullptr);
        }
        bridge->engine->destroy(bridge->controller_indirect_light);
        bridge->controller_indirect_light = nullptr;
    }
    if (bridge->scene && bridge->engine) {
        bridge->engine->destroy(bridge->scene);
    }
    if (bridge->foreground_scene && bridge->engine) {
        bridge->engine->destroy(bridge->foreground_scene);
    }
    if (bridge->asset_loader) {
        filament::gltfio::AssetLoader::destroy(&bridge->asset_loader);
    }
    if (bridge->materials) {
        bridge->materials->destroyMaterials();
        delete bridge->materials;
    }
    delete bridge->texture_provider;
    if (bridge->engine) {
        filament::Engine::destroy(&bridge->engine);
    }
    delete bridge->platform;
    delete bridge;
}

int bridge_context_wait_for_idle(FilamentBridge* bridge) {
    if (!bridge || !bridge->engine) return 0;
    bridge->engine->flushAndWait();
    return 1;
}

const char* bridge_context_last_error(const FilamentBridge* bridge) {
    return bridge ? bridge->last_error.c_str() : "bridge is null";
}
