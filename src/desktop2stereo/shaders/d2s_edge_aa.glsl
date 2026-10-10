// Image-space FXAA: encoded RGB edge detection, positive linear-light resolve.
// aa_fetch reads an integer eye-local coordinate; each wrapper clamps to its eye.
vec3 aa_decode(vec3 c) {
    return mix(c / 12.92, pow((max(c, vec3(0.0)) + 0.055) / 1.055, vec3(2.4)),
               greaterThan(c, vec3(0.04045)));
}
vec3 aa_encode(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055,
               greaterThan(c, vec3(0.0031308)));
}
float aa_luma(vec3 c) {
#if AA_LINEAR_SOURCE
    c = aa_encode(c);
#endif
    return dot(c, vec3(0.299, 0.587, 0.114));
}
float aa_luma_at(ivec2 p, uint eye) { return aa_luma(aa_fetch(p, eye)); }
float aa_sample_luma(vec2 p, uint eye) {
    ivec2 base = ivec2(floor(p));
    vec2 f = fract(p);
    return mix(mix(aa_luma_at(base, eye), aa_luma_at(base + ivec2(1, 0), eye), f.x),
               mix(aa_luma_at(base + ivec2(0, 1), eye), aa_luma_at(base + ivec2(1, 1), eye), f.x), f.y);
}
vec3 aa_sample_color(vec2 p, uint eye) {
    ivec2 base = ivec2(floor(p));
    vec2 f = fract(p);
    vec3 a = aa_fetch(base, eye), b = aa_fetch(base + ivec2(1, 0), eye);
    vec3 c = aa_fetch(base + ivec2(0, 1), eye), d = aa_fetch(base + ivec2(1, 1), eye);
#if !AA_LINEAR_SOURCE
    a = aa_decode(a); b = aa_decode(b); c = aa_decode(c); d = aa_decode(d);
#endif
    vec3 resolved = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
#if !AA_LINEAR_SOURCE
    resolved = aa_encode(resolved);
#endif
    return resolved;
}
vec3 aa_resolve(ivec2 p, uint eye) {
    vec3 original = aa_fetch(p, eye);
    float m = aa_luma(original);
    float n = aa_luma_at(p + ivec2(0, -1), eye), s = aa_luma_at(p + ivec2(0, 1), eye);
    float w = aa_luma_at(p + ivec2(-1, 0), eye), e = aa_luma_at(p + ivec2(1, 0), eye);
    float nw = aa_luma_at(p + ivec2(-1, -1), eye), ne = aa_luma_at(p + ivec2(1, -1), eye);
    float sw = aa_luma_at(p + ivec2(-1, 1), eye), se = aa_luma_at(p + ivec2(1, 1), eye);
    float low = min(m, min(min(n, s), min(w, e)));
    float high = max(m, max(max(n, s), max(w, e)));
    float range = high - low;
    if (range < max(0.0312, high * 0.125) + 1.0e-5) return original;
    float horizontal = abs(n + s - 2.0 * m) * 2.0
        + abs(ne + se - 2.0 * e) + abs(nw + sw - 2.0 * w);
    float vertical = abs(w + e - 2.0 * m) * 2.0
        + abs(nw + ne - 2.0 * n) + abs(sw + se - 2.0 * s);
    bool along_x = horizontal >= vertical - 1.0e-6;
    float a = along_x ? n : w, b = along_x ? s : e;
    float ga = abs(a - m), gb = abs(b - m);
    float sign_normal = ga >= gb - 1.0e-6 ? -1.0 : 1.0;
    float gradient = max(ga, gb);
    float edge_luma = (m + (ga >= gb ? a : b)) * 0.5;
    vec2 normal = along_x ? vec2(0.0, sign_normal) : vec2(sign_normal, 0.0);
    vec2 tangent = along_x ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec2 origin = vec2(p) + normal * 0.5;
    float distance_a = 0.0, distance_b = 0.0, delta_a = 0.0, delta_b = 0.0;
    bool done_a = false, done_b = false;
    const float steps[7] = float[7](1.0, 1.5, 2.0, 2.0, 2.0, 4.0, 8.0);
    for (int i = 0; i < 7; ++i) {
        if (!done_a) {
            distance_a += steps[i];
            delta_a = aa_sample_luma(origin - tangent * distance_a, eye) - edge_luma;
            done_a = abs(delta_a) >= gradient * 0.25 - 1.1e-4;
        }
        if (!done_b) {
            distance_b += steps[i];
            delta_b = aa_sample_luma(origin + tangent * distance_b, eye) - edge_luma;
            done_b = abs(delta_b) >= gradient * 0.25 - 1.1e-4;
        }
        if (done_a && done_b) break;
    }
    bool nearest_a = distance_a <= distance_b;
    float near_delta = nearest_a ? delta_a : delta_b;
    bool valid_span = (near_delta < 0.0) != (m - edge_luma < 0.0);
    float edge_offset = valid_span
        ? 0.5 - min(distance_a, distance_b) / max(distance_a + distance_b, 1.0e-6) : 0.0;
    float mean_luma = (2.0 * (n + s + e + w) + nw + ne + sw + se) / 12.0;
    float subpixel = clamp(abs(mean_luma - m) / max(range, 1.0e-6), 0.0, 1.0);
    subpixel = subpixel * subpixel * (3.0 - 2.0 * subpixel);
    subpixel = subpixel * subpixel * 1.25;
    float offset = max(edge_offset, subpixel);
    if (offset <= 0.0) return original;
    return aa_sample_color(vec2(p) + normal * offset, eye);
}
