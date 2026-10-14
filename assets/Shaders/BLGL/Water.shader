// Nile / sea surface.
//
// Two vertex waves running at different speeds and directions give the surface
// real movement rather than a scrolling texture. The fragment stage adds a
// fresnel term so the water is dark when viewed steeply and picks up the sky
// colour toward the horizon, plus a fine sparkle from the same wave function.

shader_type spatial;
render_mode world_vertex_coords, cull_disabled, blend_mix;

uniform vec3 shallow_color : source_color = vec3(0.16, 0.38, 0.48);
uniform vec3 deep_color : source_color = vec3(0.05, 0.16, 0.26);
uniform vec3 sky_color : source_color = vec3(0.62, 0.74, 0.86);
uniform float wave_height = 0.55;
uniform float wave_scale = 0.020;
uniform float wave_speed = 0.85;
uniform float ripple_strength = 0.35;

varying float v_wave;
varying vec3 v_world;

float wave_sum(vec2 p, float t) {
    float w = 0.0;
    w += sin(p.x * 1.00 + p.y * 0.35 + t * 1.00) * 1.00;
    w += sin(p.x * -0.42 + p.y * 0.91 + t * 1.37) * 0.62;
    w += sin(p.x * 0.17 + p.y * -0.63 + t * 0.71) * 0.38;
    return w;
}

void vertex() {
    vec2 p = VERTEX.xz * wave_scale;
    float t = TIME * wave_speed;
    float w = wave_sum(p, t);
    VERTEX.y += w * wave_height;
    v_wave = w;
    v_world = VERTEX;
}

void fragment() {
    // slope of the wave field, used as a cheap normal perturbation
    vec2 p = v_world.xz * wave_scale;
    float t = TIME * wave_speed;
    float dx = wave_sum(p + vec2(0.06, 0.0), t) - wave_sum(p - vec2(0.06, 0.0), t);
    float dz = wave_sum(p + vec2(0.0, 0.06), t) - wave_sum(p - vec2(0.0, 0.06), t);
    vec3 n = normalize(vec3(-dx * ripple_strength, 1.0, -dz * ripple_strength));

    // fresnel against the view direction
    vec3 view = normalize(CAMERA_POSITION_WORLD - v_world);
    float fres = pow(1.0 - clamp(dot(n, view), 0.0, 1.0), 3.0);

    vec3 body = mix(deep_color, shallow_color, clamp(v_wave * 0.28 + 0.5, 0.0, 1.0));
    vec3 col = mix(body, sky_color, clamp(fres * 0.85, 0.0, 1.0));

    // crest sparkle
    float crest = smoothstep(1.15, 1.75, v_wave);
    col += vec3(0.30, 0.34, 0.36) * crest * 0.6;

    ALBEDO = col;
    ALPHA = 0.93;
}
