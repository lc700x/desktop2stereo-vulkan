#version 450

layout(set = 0, binding = 0) uniform sampler2D screen_texture;
layout(set = 0, binding = 1, std430) readonly buffer ScreenCropState {
    vec4 source_crop;
} crop_state;
layout(push_constant) uniform ScreenParams {
    mat4 view_projection;
    vec4 center;
    vec4 right;
    vec4 up;
    vec4 size_curve;
} params;

layout(location = 0) in vec2 texture_uv;
layout(location = 0) out vec4 output_color;

vec2 crop_min_uv() {
    return clamp(crop_state.source_crop.xy, vec2(0.0), vec2(0.9));
}

vec2 crop_max_uv() {
    vec2 minimum = crop_min_uv();
    return clamp(
        minimum + max(crop_state.source_crop.zw, vec2(0.1)),
        minimum + vec2(0.1), vec2(1.0)
    );
}

vec2 crop_source_uv(vec2 display_uv) {
    return mix(crop_min_uv(), crop_max_uv(), clamp(display_uv, vec2(0.0), vec2(1.0)));
}

vec4 sample_pixel_area(vec2 uv) {
    vec2 source_min_uv = crop_min_uv();
    vec2 source_max_uv = crop_max_uv();
    vec2 source_uv = crop_source_uv(uv);
    vec2 source_dx = dFdx(source_uv);
    vec2 source_dy = dFdy(source_uv);
    vec2 source_size = vec2(textureSize(screen_texture, 0));
    float footprint_x = length(source_dx * source_size);
    float footprint_y = length(source_dy * source_size);
    if (max(footprint_x, footprint_y) <= 1.0) {
        return textureLod(screen_texture, source_uv, 0.0);
    }

    // Four-point Gauss-Legendre quadrature integrates the projected pixel's
    // footprint instead of sampling sparsely across neighboring pixels.
    const float offsets[4] = float[](
        -0.4305681558, -0.1699905218, 0.1699905218, 0.4305681558
    );
    const float weights[4] = float[](
        0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226
    );
    vec4 total = vec4(0.0);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            vec2 sample_uv = clamp(
                source_uv + source_dx * offsets[x] + source_dy * offsets[y],
                source_min_uv,
                source_max_uv
            );
            total += textureLod(screen_texture, sample_uv, 0.0)
                * weights[x] * weights[y];
        }
    }
    return total;
}

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

vec3 sample_easu_source(vec2 pixel, vec2 source_texel) {
    return texture(screen_texture, clamp(
        (pixel + vec2(0.5)) * source_texel, crop_min_uv(), crop_max_uv()
    )).rgb;
}

void easu_set(
    inout vec2 direction, inout float length_value, float weight,
    float a, float b, float c, float d, float e
) {
    float direction_x = d - b;
    float length_x = 1.0 / max(max(abs(d - c), abs(c - b)), 1e-6);
    direction.x += direction_x * weight;
    length_x = clamp(abs(direction_x) * length_x, 0.0, 1.0);
    length_value += length_x * length_x * weight;

    float direction_y = e - a;
    float length_y = 1.0 / max(max(abs(e - c), abs(c - a)), 1e-6);
    direction.y += direction_y * weight;
    length_y = clamp(abs(direction_y) * length_y, 0.0, 1.0);
    length_value += length_y * length_y * weight;
}

void easu_tap(
    inout vec3 color, inout float weight_sum, vec2 offset,
    vec2 direction, vec2 length_value, float lobe, float clip_value,
    vec3 sample_color
) {
    vec2 rotated = vec2(
        offset.x * direction.x + offset.y * direction.y,
        offset.x * -direction.y + offset.y * direction.x
    ) * length_value;
    float distance_squared = min(dot(rotated, rotated), clip_value);
    float weight_b = 0.4 * distance_squared - 1.0;
    float weight_a = lobe * distance_squared - 1.0;
    weight_b = 1.5625 * weight_b * weight_b - 0.5625;
    float weight = weight_b * weight_a * weight_a;
    color += sample_color * weight;
    weight_sum += weight;
}

vec3 sample_easu(vec2 uv) {
    vec2 source_texel = abs(vec2(params.center.w, params.right.w));
    vec2 source_size = (crop_max_uv() - crop_min_uv()) / source_texel;
    vec2 output_size = max(vec2(params.up.w, params.size_curve.w), vec2(1.0));
    vec2 output_uv = (floor(uv * output_size) + vec2(0.5)) / output_size;
    vec2 source_position = crop_min_uv() / source_texel
        + output_uv * source_size - vec2(0.5);
    vec2 source_base = floor(source_position);
    vec2 pp = source_position - source_base;

    vec3 b = sample_easu_source(source_base + vec2(0.0, -1.0), source_texel);
    vec3 c = sample_easu_source(source_base + vec2(1.0, -1.0), source_texel);
    vec3 e = sample_easu_source(source_base + vec2(-1.0, 0.0), source_texel);
    vec3 f = sample_easu_source(source_base, source_texel);
    vec3 g = sample_easu_source(source_base + vec2(1.0, 0.0), source_texel);
    vec3 h = sample_easu_source(source_base + vec2(2.0, 0.0), source_texel);
    vec3 i = sample_easu_source(source_base + vec2(-1.0, 1.0), source_texel);
    vec3 j = sample_easu_source(source_base + vec2(0.0, 1.0), source_texel);
    vec3 k = sample_easu_source(source_base + vec2(1.0, 1.0), source_texel);
    vec3 l = sample_easu_source(source_base + vec2(2.0, 1.0), source_texel);
    vec3 n = sample_easu_source(source_base + vec2(0.0, 2.0), source_texel);
    vec3 o = sample_easu_source(source_base + vec2(1.0, 2.0), source_texel);

    float b_luma = luma(b); float c_luma = luma(c); float e_luma = luma(e);
    float f_luma = luma(f); float g_luma = luma(g); float h_luma = luma(h);
    float i_luma = luma(i); float j_luma = luma(j); float k_luma = luma(k);
    float l_luma = luma(l); float n_luma = luma(n); float o_luma = luma(o);
    vec2 direction = vec2(0.0);
    float length_value = 0.0;
    easu_set(direction, length_value, (1.0 - pp.x) * (1.0 - pp.y), b_luma, e_luma, f_luma, g_luma, j_luma);
    easu_set(direction, length_value, pp.x * (1.0 - pp.y), c_luma, f_luma, g_luma, h_luma, k_luma);
    easu_set(direction, length_value, (1.0 - pp.x) * pp.y, f_luma, i_luma, j_luma, k_luma, n_luma);
    easu_set(direction, length_value, pp.x * pp.y, g_luma, j_luma, k_luma, l_luma, o_luma);
    float direction_length = dot(direction, direction);
    bool zero_direction = direction_length < 0.000030517578125;
    direction = zero_direction ? vec2(1.0, 0.0) : direction / sqrt(direction_length);
    length_value = 0.25 * length_value * length_value;
    float stretch = 1.0 / max(max(abs(direction.x), abs(direction.y)), 1e-6);
    vec2 length_squared = vec2(
        1.0 + (stretch - 1.0) * length_value,
        1.0 - 0.5 * length_value
    );
    float lobe = 0.5 + (0.21 - 0.5) * length_value;
    float clip_value = 1.0 / max(lobe, 1e-6);
    vec3 min4 = min(min(f, g), min(j, k));
    vec3 max4 = max(max(f, g), max(j, k));
    vec3 color = vec3(0.0);
    float weight_sum = 0.0;
    easu_tap(color, weight_sum, vec2(0.0, -1.0) - pp, direction, length_squared, lobe, clip_value, b);
    easu_tap(color, weight_sum, vec2(1.0, -1.0) - pp, direction, length_squared, lobe, clip_value, c);
    easu_tap(color, weight_sum, vec2(-1.0, 1.0) - pp, direction, length_squared, lobe, clip_value, i);
    easu_tap(color, weight_sum, vec2(0.0, 1.0) - pp, direction, length_squared, lobe, clip_value, j);
    easu_tap(color, weight_sum, vec2(0.0, 0.0) - pp, direction, length_squared, lobe, clip_value, f);
    easu_tap(color, weight_sum, vec2(-1.0, 0.0) - pp, direction, length_squared, lobe, clip_value, e);
    easu_tap(color, weight_sum, vec2(1.0, 1.0) - pp, direction, length_squared, lobe, clip_value, k);
    easu_tap(color, weight_sum, vec2(2.0, 1.0) - pp, direction, length_squared, lobe, clip_value, l);
    easu_tap(color, weight_sum, vec2(2.0, 0.0) - pp, direction, length_squared, lobe, clip_value, h);
    easu_tap(color, weight_sum, vec2(1.0, 0.0) - pp, direction, length_squared, lobe, clip_value, g);
    easu_tap(color, weight_sum, vec2(1.0, 2.0) - pp, direction, length_squared, lobe, clip_value, o);
    easu_tap(color, weight_sum, vec2(0.0, 2.0) - pp, direction, length_squared, lobe, clip_value, n);
    return weight_sum <= 1e-6 ? f : min(max4, max(min4, color / weight_sum));
}

void main() {
    vec4 color = params.center.w < 0.0
        ? vec4(sample_easu(texture_uv), 1.0)
        : (params.size_curve.w > 0.5
            ? sample_pixel_area(texture_uv)
            : texture(screen_texture, crop_source_uv(texture_uv)));
    output_color = vec4(color.rgb, params.size_curve.w < -0.5
        ? clamp(dot(color.rgb, vec3(0.299, 0.587, 0.114)) * 0.35, 0.0, 0.35)
        : 1.0);
}
