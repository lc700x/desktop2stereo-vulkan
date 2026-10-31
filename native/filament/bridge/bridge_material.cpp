#include "bridge_material.h"
#include "bridge_internal.h"
#include "bridge_eye.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

bool valid_color(const float color[3]) {
    return std::isfinite(color[0]) && std::isfinite(color[1]) &&
            std::isfinite(color[2]) && color[0] >= 0.0f &&
            color[1] >= 0.0f && color[2] >= 0.0f;
}

bool color_visible(const float color[3]) {
    return color[0] > 0.0f || color[1] > 0.0f || color[2] > 0.0f;
}

bool valid_vec3(const float value[3]) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) &&
            std::isfinite(value[2]);
}

}  // namespace

template<typename Target>
bool configure_color_pipeline_impl(Target* target) {
    if (!target || !target->engine || !target->view) {
        return false;
    }
    auto* previous = target->color_grading;
    target->color_grading = filament::ColorGrading::Builder()
            .toneMapping(filament::ColorGrading::ToneMapping::LINEAR)
            .exposure(target->brightness.scene_exposure_ev)
            // Keep the projection target in sRGB format and let its target
            // conversion perform the single sRGB OETF at store time.
            .outputColorSpace(filament::color::Rec709 - filament::color::Linear - filament::color::D65)
            .build(*target->engine);
    if (!target->color_grading) {
        return false;
    }
    target->view->setColorGrading(target->color_grading);
    target->view->setPostProcessingEnabled(true);
    if (previous) {
        target->engine->destroy(previous);
    }
    return true;
}

bool is_skybox_name(const char* name) {
    if (!name) return false;
    std::string value(name);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value.find("skybox") != std::string::npos;
}

template<typename BridgeType>
void apply_material_brightness_impl(BridgeType* bridge) {
    if (!bridge || !bridge->engine) return;
    const float skybox_factor = bridge->brightness.skybox_brightness;
    for (const auto& entry : bridge->brightness.skybox_materials) {
        if (!entry.material) continue;
        const auto& base = entry.base_color_factor;
        entry.material->setParameter("baseColorFactor", filament::math::float4{
                base.x * skybox_factor, base.y * skybox_factor,
                base.z * skybox_factor, base.w});
    }
}

template<typename BridgeType>
void collect_material_brightness_impl(BridgeType* bridge, bool enable_fill_channel) {
    if (!bridge || !bridge->engine || !bridge->asset) return;
    auto& renderables = bridge->engine->getRenderableManager();
    bridge->brightness.scene_materials.clear();
    bridge->brightness.skybox_materials.clear();
    bridge->brightness.skybox_entities.clear();
    const auto* entities = bridge->asset->getRenderableEntities();
    for (size_t index = 0; index < bridge->asset->getRenderableEntityCount(); ++index) {
        const utils::Entity entity = entities[index];
        auto instance = renderables.getInstance(entity);
        if (!instance.isValid()) continue;
        const bool skybox = is_skybox_name(bridge->asset->getName(entity));
        if (skybox) {
            // Render the exported skybox before regular scene geometry so its
            // depth buffer cannot hide the Saturn ring or other scene meshes.
            renderables.setPriority(instance, 0);
            renderables.setLightChannel(instance, 0, false);
            renderables.setLightChannel(instance, 1, false);
            bridge->brightness.skybox_entities.push_back(entity);
        } else if (enable_fill_channel) {
            renderables.setLightChannel(instance, 1, true);
        }
        auto& target = skybox
                ? bridge->brightness.skybox_materials
                : bridge->brightness.scene_materials;
        for (size_t primitive = 0; primitive < renderables.getPrimitiveCount(instance); ++primitive) {
            auto* material = renderables.getMaterialInstanceAt(instance, primitive);
            if (!material || !material->getMaterial()->hasParameter("baseColorFactor")) continue;
            target.push_back({material, material->template getParameter<filament::math::float4>(
                    "baseColorFactor")});
        }
    }
    apply_material_brightness_impl(bridge);
}

int preview_material_set_scene_exposure(FilamentPreview* preview, float exposure_ev) {
    if (!preview || !preview->engine || !std::isfinite(exposure_ev)) {
        return 0;
    }
    preview->brightness.scene_exposure_ev = std::clamp(exposure_ev, -8.0f, 8.0f);
    return configure_color_pipeline_impl(preview) ? 1 : 0;
}

int preview_material_set_fill_light(
        FilamentPreview* preview,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z) {
    if (!preview || !preview->engine || !preview->scene ||
            !std::isfinite(red) || !std::isfinite(green) || !std::isfinite(blue) ||
            !std::isfinite(intensity) || intensity < 0.0f ||
            !std::isfinite(direction_x) || !std::isfinite(direction_y) ||
            !std::isfinite(direction_z)) {
        return 0;
    }
    if (!preview->fill_light.isNull()) {
        preview->scene->remove(preview->fill_light);
        preview->engine->destroy(preview->fill_light);
    }
    preview->fill_light = utils::EntityManager::get().create();
    filament::LightManager::Builder(filament::LightManager::Type::DIRECTIONAL)
            .color(filament::LinearColor{red, green, blue})
            .intensity(intensity)
            .direction({direction_x, direction_y, direction_z})
            .lightChannel(0, false)
            .lightChannel(1, true)
            .castShadows(false)
            .build(*preview->engine, preview->fill_light);
    preview->scene->addEntity(preview->fill_light);
    return 1;
}

int preview_material_set_ambient_light(
        FilamentPreview* preview, float red, float green, float blue) {
    if (!preview || !preview->engine || !preview->scene ||
            !std::isfinite(red) || !std::isfinite(green) ||
            !std::isfinite(blue) || red < 0.0f || green < 0.0f || blue < 0.0f) {
        return 0;
    }
    filament::IndirectLight* next = nullptr;
    if (red > 0.0f || green > 0.0f || blue > 0.0f) {
        const filament::math::float3 irradiance[] = {{red, green, blue}};
        next = filament::IndirectLight::Builder()
                .irradiance(1, irradiance)
                .intensity(preview->ambient_intensity_lux)
                .build(*preview->engine);
        if (!next) return 0;
    }
    auto* previous = preview->indirect_light;
    preview->scene->setIndirectLight(next);
    preview->indirect_light = next;
    if (previous) preview->engine->destroy(previous);
    return 1;
}

int preview_material_set_skybox_brightness(FilamentPreview* preview, float brightness) {
    if (!preview || !std::isfinite(brightness) || brightness < 0.0f) return 0;
    preview->brightness.skybox_brightness = std::min(brightness, 16.0f);
    bridge_material_apply_brightness(preview);
    return 1;
}

int bridge_material_set_scene_exposure(FilamentBridge* bridge, float exposure_ev) {
    if (!bridge || !std::isfinite(exposure_ev)) return 0;
    bridge->brightness.scene_exposure_ev = std::clamp(exposure_ev, -8.0f, 8.0f);
    const uint32_t active_eye = bridge->active_eye;
    for (uint32_t eye_index = 0; eye_index < bridge->eyes.size(); ++eye_index) {
        auto& eye = bridge->eyes[eye_index];
        if (!eye.view) continue;
        bridge_eye_activate(bridge, eye_index);
        // The foreground view must not retain the grading object while the
        // main view replaces it during an exposure update.
        if (eye.foreground_view) {
            eye.foreground_view->setColorGrading(nullptr);
        }
        bridge->color_grading = eye.color_grading;
        if (!configure_color_pipeline_impl(bridge)) {
            if (eye.foreground_view) {
                eye.foreground_view->setColorGrading(eye.color_grading);
            }
            bridge_eye_activate(bridge, active_eye);
            return 0;
        }
        eye.color_grading = bridge->color_grading;
        if (eye.foreground_view) {
            eye.foreground_view->setColorGrading(eye.color_grading);
        }
        // controller_view and controller_guide_view keep their dedicated
        // neutral grading object. Only their own ambient/head/top/screen
        // lights may affect PBR controllers; the Unlit laser bypasses lights.
    }
    bridge_eye_activate(bridge, active_eye);
    return 1;
}

int bridge_material_set_skybox_brightness(FilamentBridge* bridge, float brightness) {
    if (!bridge || !std::isfinite(brightness) || brightness < 0.0f) return 0;
    bridge->brightness.skybox_brightness = std::min(brightness, 16.0f);
    apply_material_brightness_impl(bridge);
    return 1;
}

int bridge_material_set_ambient_light(
        FilamentBridge* bridge, float red, float green, float blue) {
    if (!bridge) return 0;
    auto config = bridge->lighting;
    config.struct_size = sizeof(config);
    config.environment_ambient_color[0] = red;
    config.environment_ambient_color[1] = green;
    config.environment_ambient_color[2] = blue;
    return bridge_material_set_lighting_config(bridge, &config);
}

int preview_material_set_ambient_light_with_intensity(
        FilamentPreview* preview, float red, float green, float blue,
        float intensity_lux) {
    if (!preview || !std::isfinite(intensity_lux) || intensity_lux < 0.0f) {
        return 0;
    }
    preview->ambient_intensity_lux = intensity_lux;
    return preview_material_set_ambient_light(preview, red, green, blue);
}

int bridge_material_set_controller_ambient_light(
        FilamentBridge* bridge, float red, float green, float blue, int enabled) {
    if (!bridge) return 0;
    auto config = bridge->lighting;
    config.struct_size = sizeof(config);
    config.controller_ambient_color[0] = red;
    config.controller_ambient_color[1] = green;
    config.controller_ambient_color[2] = blue;
    config.controller_ambient_enabled = enabled;
    return bridge_material_set_lighting_config(bridge, &config);
}

int bridge_material_set_lighting_config(
        FilamentBridge* bridge, const FilamentBridgeLightingConfig* config) {
    if (!bridge || !bridge->engine || !bridge->scene || !bridge->foreground_scene ||
            !config || config->struct_size < sizeof(FilamentBridgeLightingConfig) ||
            !valid_color(config->environment_ambient_color) ||
            !valid_color(config->controller_ambient_color) ||
            !valid_color(config->head_light_color) ||
            !valid_color(config->top_light_color) ||
            !valid_vec3(config->head_light_offset) ||
            !valid_vec3(config->top_light_offset) ||
            !std::isfinite(config->environment_ambient_intensity_lux) ||
            config->environment_ambient_intensity_lux < 0.0f ||
            !std::isfinite(config->controller_ambient_intensity_lux) ||
            config->controller_ambient_intensity_lux < 0.0f ||
            !std::isfinite(config->head_light_intensity_candela) ||
            config->head_light_intensity_candela < 0.0f ||
            !std::isfinite(config->top_light_intensity_candela) ||
            config->top_light_intensity_candela < 0.0f ||
            !std::isfinite(config->head_light_falloff) ||
            config->head_light_falloff <= 0.0f ||
            !std::isfinite(config->top_light_falloff) ||
            config->top_light_falloff <= 0.0f) {
        return 0;
    }

    filament::IndirectLight* environment_ambient = nullptr;
    if (config->environment_ambient_intensity_lux > 0.0f &&
            color_visible(config->environment_ambient_color)) {
        const filament::math::float3 irradiance[] = {{
                config->environment_ambient_color[0],
                config->environment_ambient_color[1],
                config->environment_ambient_color[2]}};
        environment_ambient = filament::IndirectLight::Builder()
                .irradiance(1, irradiance)
                .intensity(config->environment_ambient_intensity_lux)
                .build(*bridge->engine);
        if (!environment_ambient) return 0;
    }
    filament::IndirectLight* controller_ambient = nullptr;
    if (config->controller_ambient_enabled &&
            config->controller_ambient_intensity_lux > 0.0f &&
            color_visible(config->controller_ambient_color)) {
        const filament::math::float3 irradiance[] = {{
                config->controller_ambient_color[0],
                config->controller_ambient_color[1],
                config->controller_ambient_color[2]}};
        controller_ambient = filament::IndirectLight::Builder()
                .irradiance(1, irradiance)
                .intensity(config->controller_ambient_intensity_lux)
                .build(*bridge->engine);
        if (!controller_ambient) {
            if (environment_ambient) bridge->engine->destroy(environment_ambient);
            return 0;
        }
    }

    if (!bridge->fill_light.isNull()) {
        bridge->scene->remove(bridge->fill_light);
        bridge->foreground_scene->remove(bridge->fill_light);
        bridge->engine->destroy(bridge->fill_light);
        bridge->fill_light = {};
    }
    if (!bridge->controller_top_light.isNull()) {
        bridge->scene->remove(bridge->controller_top_light);
        bridge->foreground_scene->remove(bridge->controller_top_light);
        bridge->engine->destroy(bridge->controller_top_light);
        bridge->controller_top_light = {};
    }
    if (config->head_light_intensity_candela > 0.0f &&
            color_visible(config->head_light_color)) {
        bridge->fill_light = utils::EntityManager::get().create();
        filament::LightManager::Builder(filament::LightManager::Type::POINT)
                .color(filament::LinearColor{
                        config->head_light_color[0], config->head_light_color[1],
                        config->head_light_color[2]})
                .intensityCandela(config->head_light_intensity_candela)
                .position({config->head_light_offset[0], config->head_light_offset[1],
                        config->head_light_offset[2]})
                .falloff(config->head_light_falloff)
                .lightChannel(0, false).lightChannel(1, true)
                .castShadows(config->head_light_cast_shadows != 0)
                .build(*bridge->engine, bridge->fill_light);
        bridge->foreground_scene->addEntity(bridge->fill_light);
    }
    if (config->top_light_intensity_candela > 0.0f &&
            color_visible(config->top_light_color)) {
        bridge->controller_top_light = utils::EntityManager::get().create();
        filament::LightManager::Builder(filament::LightManager::Type::POINT)
                .color(filament::LinearColor{
                        config->top_light_color[0], config->top_light_color[1],
                        config->top_light_color[2]})
                .intensityCandela(config->top_light_intensity_candela)
                .position({config->top_light_offset[0], config->top_light_offset[1],
                        config->top_light_offset[2]})
                .falloff(config->top_light_falloff)
                .lightChannel(0, false).lightChannel(1, true)
                .castShadows(config->top_light_cast_shadows != 0)
                .build(*bridge->engine, bridge->controller_top_light);
        bridge->foreground_scene->addEntity(bridge->controller_top_light);
    }

    auto* previous_environment = bridge->indirect_light;
    auto* previous_controller = bridge->controller_indirect_light;
    bridge->scene->setIndirectLight(environment_ambient);
    bridge->foreground_scene->setIndirectLight(controller_ambient);
    bridge->indirect_light = environment_ambient;
    bridge->controller_indirect_light = controller_ambient;
    if (previous_environment) bridge->engine->destroy(previous_environment);
    if (previous_controller) bridge->engine->destroy(previous_controller);
    bridge->lighting = *config;
    return 1;
}

int bridge_material_set_controller_screen_light(
        FilamentBridge* bridge, float red, float green, float blue,
        float intensity_lux, float direction_x, float direction_y,
        float direction_z, int cast_shadows, int enabled) {
    const float color[] = {red, green, blue};
    const bool active = enabled != 0 && intensity_lux > 0.0f &&
            color_visible(color);
    if (!bridge || !bridge->engine || !bridge->foreground_scene ||
            !valid_color(color) || !std::isfinite(intensity_lux) ||
            intensity_lux < 0.0f || !std::isfinite(direction_x) ||
            !std::isfinite(direction_y) || !std::isfinite(direction_z)) {
        return 0;
    }
    if (!active) {
        if (!bridge->controller_screen_light.isNull()) {
            bridge->scene->remove(bridge->controller_screen_light);
            bridge->foreground_scene->remove(bridge->controller_screen_light);
            bridge->engine->destroy(bridge->controller_screen_light);
            bridge->controller_screen_light = {};
        }
        return 1;
    }

    const bool shadow_enabled = cast_shadows != 0;
    if (!bridge->controller_screen_light.isNull() &&
            bridge->controller_screen_light_cast_shadows != shadow_enabled) {
        bridge->scene->remove(bridge->controller_screen_light);
        bridge->foreground_scene->remove(bridge->controller_screen_light);
        bridge->engine->destroy(bridge->controller_screen_light);
        bridge->controller_screen_light = {};
    }
    if (bridge->controller_screen_light.isNull()) {
        bridge->controller_screen_light = utils::EntityManager::get().create();
        filament::LightManager::Builder(filament::LightManager::Type::DIRECTIONAL)
                .color(filament::LinearColor{red, green, blue})
                .intensity(intensity_lux)
                .direction({direction_x, direction_y, direction_z})
                .lightChannel(0, false).lightChannel(1, true)
                .castShadows(shadow_enabled)
                .build(*bridge->engine, bridge->controller_screen_light);
        bridge->foreground_scene->addEntity(bridge->controller_screen_light);
        bridge->controller_screen_light_cast_shadows = shadow_enabled;
        return 1;
    }

    auto& lights = bridge->engine->getLightManager();
    const auto instance = lights.getInstance(bridge->controller_screen_light);
    if (!instance.isValid()) return 0;
    lights.setColor(instance, filament::LinearColor{red, green, blue});
    lights.setIntensity(instance, intensity_lux);
    lights.setDirection(instance, {direction_x, direction_y, direction_z});
    return 1;
}

int bridge_material_set_environment_screen_lights(
        FilamentBridge* bridge, const float* positions_xyz,
        const float* linear_rgb, const float* intensity_candela,
        uint32_t count, float falloff, int cast_shadows, int enabled) {
    constexpr uint32_t kMaximumLights = 24;
    if (!bridge || !bridge->engine || !bridge->scene ||
            !std::isfinite(falloff) || falloff <= 0.0f ||
            count > kMaximumLights ||
            (enabled != 0 && count > 0 &&
                    (!positions_xyz || !linear_rgb || !intensity_candela))) {
        return 0;
    }
    const bool active = enabled != 0 && count > 0;
    const bool shadow_enabled = cast_shadows != 0;
    if (active) {
        for (uint32_t index = 0; index < count; ++index) {
            const float* position = positions_xyz + index * 3;
            const float* color = linear_rgb + index * 3;
            const float intensity = intensity_candela[index];
            if (!valid_vec3(position) || !valid_color(color) ||
                    !std::isfinite(intensity) || intensity < 0.0f) {
                return 0;
            }
        }
    }
    const bool recreate = active && (
            bridge->environment_screen_light_count != count ||
            bridge->environment_screen_light_cast_shadows != shadow_enabled ||
            std::abs(bridge->environment_screen_light_falloff - falloff) > 1e-5f);
    if (!active || recreate) {
        for (auto& light : bridge->environment_screen_lights) {
            if (!light.isNull()) {
                bridge->scene->remove(light);
                bridge->engine->destroy(light);
                light = {};
            }
        }
        bridge->environment_screen_light_count = 0;
    }
    if (!active) return 1;

    auto& lights = bridge->engine->getLightManager();
    for (uint32_t index = 0; index < count; ++index) {
        const float* position = positions_xyz + index * 3;
        const float* color = linear_rgb + index * 3;
        const float intensity = intensity_candela[index];
        auto& light = bridge->environment_screen_lights[index];
        if (light.isNull()) {
            light = utils::EntityManager::get().create();
            filament::LightManager::Builder(filament::LightManager::Type::POINT)
                    .color(filament::LinearColor{color[0], color[1], color[2]})
                    .intensityCandela(intensity)
                    .position({position[0], position[1], position[2]})
                    .falloff(falloff)
                    .lightChannel(0, true).lightChannel(1, false)
                    .castShadows(shadow_enabled)
                    .build(*bridge->engine, light);
            bridge->scene->addEntity(light);
            continue;
        }
        const auto instance = lights.getInstance(light);
        if (!instance.isValid()) return 0;
        lights.setColor(instance, filament::LinearColor{
                color[0], color[1], color[2]});
        lights.setIntensityCandela(instance, intensity);
        lights.setPosition(instance, {position[0], position[1], position[2]});
    }
    bridge->environment_screen_light_count = count;
    bridge->environment_screen_light_falloff = falloff;
    bridge->environment_screen_light_cast_shadows = shadow_enabled;
    if (recreate) {
        float maximum_intensity = 0.0f;
        for (uint32_t index = 0; index < count; ++index) {
            maximum_intensity = std::max(
                    maximum_intensity, intensity_candela[index]);
        }
        std::fprintf(stderr,
                "[FilamentBridge] room screen lights active: count=%u "
                "max_cd=%.3f falloff=%.3f first_position=(%.3f,%.3f,%.3f)\n",
                count, maximum_intensity, falloff,
                positions_xyz[0], positions_xyz[1], positions_xyz[2]);
    }
    return 1;
}

int bridge_material_set_environment_screen_area_light(
        FilamentBridge* bridge, const float* position,
        const float* normal, const float* color,
        float half_width, float half_height, float intensity, int enabled) {
    if (!bridge || !bridge->engine ||
            (enabled != 0 && (!valid_vec3(position) || !valid_vec3(normal) ||
                    !valid_color(color) || !std::isfinite(half_width) ||
                    !std::isfinite(half_height) || !std::isfinite(intensity) ||
                    half_width <= 0.0f || half_height <= 0.0f || intensity < 0.0f))) {
        return 0;
    }
    const bool active = enabled != 0 && intensity > 0.0f;
    const filament::math::float4 globals[] = {
        active ? filament::math::float4{position[0], position[1], position[2], 1.0f}
               : filament::math::float4{0.0f, 0.0f, 0.0f, 0.0f},
        active ? filament::math::float4{normal[0], normal[1], normal[2], intensity}
               : filament::math::float4{0.0f},
        active ? filament::math::float4{color[0], color[1], color[2], half_width}
               : filament::math::float4{0.0f},
        active ? filament::math::float4{half_height, 0.0f, 0.0f, 0.0f}
               : filament::math::float4{0.0f},
    };
    for (auto& eye : bridge->eyes) {
        if (!eye.view) continue;
        for (uint32_t index = 0; index < 4; ++index) {
            eye.view->setMaterialGlobal(index, globals[index]);
        }
    }
    // The area-light material path replaces the old punctual-light
    // approximation. Remove any legacy point lights when this ABI is used.
    return bridge_material_set_environment_screen_lights(
            bridge, nullptr, nullptr, nullptr, 0, 1.0f, 0, 0);
}

int bridge_material_set_fill_light(
        FilamentBridge* bridge,
        float red, float green, float blue,
        float intensity,
        float direction_x, float direction_y, float direction_z) {
    if (!bridge || !std::isfinite(red) || !std::isfinite(green) ||
            !std::isfinite(blue) || !std::isfinite(intensity) || intensity < 0.0f ||
            !std::isfinite(direction_x) || !std::isfinite(direction_y) ||
            !std::isfinite(direction_z)) return 0;
    auto config = bridge->lighting;
    config.struct_size = sizeof(config);
    config.head_light_color[0] = red;
    config.head_light_color[1] = green;
    config.head_light_color[2] = blue;
    config.head_light_intensity_candela = intensity;
    return bridge_material_set_lighting_config(bridge, &config);
}

int bridge_material_set_passthrough_backdrop(
        FilamentBridge* bridge, int enabled) {
    if (!bridge || !bridge->engine) return 0;
    const bool active = enabled != 0;
    bridge->passthrough_backdrop = active;
    auto& renderables = bridge->engine->getRenderableManager();
    for (const auto entity : bridge->brightness.skybox_entities) {
        const auto instance = renderables.getInstance(entity);
        if (instance.isValid()) {
            renderables.setLayerMask(instance, 0xff, active ? 0x00 : 0x01);
        }
    }
    for (auto& eye : bridge->eyes) {
        if (!eye.renderer) continue;
        filament::Renderer::ClearOptions clear_options;
        // Store the requested sRGB chroma key RGB (0, 153, 51) as linear
        // clear values for the OpenXR sRGB swapchain.
        clear_options.clearColor = active
                ? filament::math::float4{
                        0.0f, 0.31854678f, 0.03310477f, 1.0f}
                : filament::math::float4{0.0f, 0.0f, 0.0f, 1.0f};
        clear_options.clear = true;
        clear_options.discard = true;
        eye.renderer->setClearOptions(clear_options);
    }
    return 1;
}

void bridge_material_update_controller_lights(
        FilamentBridge* bridge, float eye_x, float eye_y, float eye_z) {
    if (!bridge || !bridge->engine) return;
    auto& lights = bridge->engine->getLightManager();
    if (!bridge->fill_light.isNull()) {
        const auto instance = lights.getInstance(bridge->fill_light);
        if (instance.isValid()) {
            lights.setPosition(instance, {
                    eye_x + bridge->lighting.head_light_offset[0],
                    eye_y + bridge->lighting.head_light_offset[1],
                    eye_z + bridge->lighting.head_light_offset[2]});
        }
    }
    if (!bridge->controller_top_light.isNull()) {
        const auto instance = lights.getInstance(bridge->controller_top_light);
        if (instance.isValid()) {
            lights.setPosition(instance, {
                    eye_x + bridge->lighting.top_light_offset[0],
                    eye_y + bridge->lighting.top_light_offset[1],
                    eye_z + bridge->lighting.top_light_offset[2]});
        }
    }
}

bool bridge_material_configure_color_pipeline(FilamentBridge* bridge) {
    return configure_color_pipeline_impl(bridge);
}

bool bridge_material_configure_color_pipeline(FilamentPreview* preview) {
    return configure_color_pipeline_impl(preview);
}

void bridge_material_collect_brightness(
        FilamentBridge* bridge, bool enable_fill_channel) {
    collect_material_brightness_impl(bridge, enable_fill_channel);
}

void bridge_material_collect_brightness(
        FilamentPreview* preview, bool enable_fill_channel) {
    collect_material_brightness_impl(preview, enable_fill_channel);
}

void bridge_material_apply_brightness(FilamentBridge* bridge) {
    apply_material_brightness_impl(bridge);
}

void bridge_material_apply_brightness(FilamentPreview* preview) {
    apply_material_brightness_impl(preview);
}
