#version 450

layout(set = 0, binding = 1, std430) readonly buffer ScreenCropState {
    vec4 source_crop;
    vec4 target_viewport;
} crop_state;

layout(push_constant) uniform ScreenParams {
    mat4 view_projection;
    vec4 center;
    vec4 right;
    vec4 up;
    vec4 size_curve;
} params;

layout(location = 0) out vec2 texture_uv;

const int SEGMENTS = 48;

vec3 screen_position(vec2 uv) {
    float local_x;
    float local_z = 0.0;
    float half_angle = params.size_curve.z;
    if (half_angle > 0.0) {
        float angle = mix(-half_angle, half_angle, uv.x);
        float radius = params.size_curve.x / half_angle;
        local_x = radius * sin(angle);
        local_z = radius * (1.0 - cos(angle));
    } else {
        local_x = mix(-params.size_curve.x, params.size_curve.x, uv.x);
    }
    float local_y = mix(params.size_curve.y, -params.size_curve.y, uv.y);
    vec3 forward = normalize(cross(params.right.xyz, params.up.xyz));
    return params.center.xyz
        + params.right.xyz * local_x
        + params.up.xyz * local_y
        + forward * local_z;
}

void main() {
    int column = gl_VertexIndex / 2;
    bool top = (gl_VertexIndex & 1) != 0;
    vec2 uv = vec2(float(column) / float(SEGMENTS), top ? 0.0 : 1.0);
    vec4 clip = params.view_projection * vec4(screen_position(uv), 1.0);
    if (clip.w > 1e-3) {
        float angle = mix(-params.size_curve.z, params.size_curve.z, uv.x);
        vec3 forward = normalize(cross(params.right.xyz, params.up.xyz));
        vec3 tangent_u = 2.0 * params.size_curve.x
            * (params.right.xyz * cos(angle) + forward * sin(angle));
        vec3 tangent_v = -2.0 * params.size_curve.y * params.up.xyz;
        vec4 du = params.view_projection * vec4(tangent_u, 0.0);
        vec4 dv = params.view_projection * vec4(tangent_v, 0.0);
        vec2 viewport = max(crop_state.target_viewport.xy, vec2(1.0));
        vec2 pixel_du = (du.xy * clip.w - clip.xy * du.w)
            / (clip.w * clip.w) * viewport * 0.5;
        vec2 pixel_dv = (dv.xy * clip.w - clip.xy * dv.w)
            / (clip.w * clip.w) * viewport * 0.5;
        float determinant = abs(pixel_du.x * pixel_dv.y - pixel_du.y * pixel_dv.x);
        vec2 fringe = min(
            1.5 * vec2(length(pixel_dv), length(pixel_du)) / max(determinant, 1e-6),
            vec2(0.25)
        );
        // Rasterize beyond the silhouette so the fragment shader can cover
        // both halves of the edge pixel instead of clipping its outer half.
        if (column == 0) uv.x -= fringe.x;
        if (column == SEGMENTS) uv.x += fringe.x;
        uv.y += top ? -fringe.y : fringe.y;
    }
    gl_Position = params.view_projection * vec4(screen_position(uv), 1.0);
    texture_uv = uv;
}
