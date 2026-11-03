"""GLSL 330 sources. All world units are centimeters."""

# Rounded card corners (material.corner_radius_cm): true outside the rounded rectangle, in page centimeters.
# The page passes set u_corner_radius; everything else draws with 0, which keeps every fragment.
_CORNER = """
uniform float u_corner_radius;
uniform vec2 u_page_size;
bool outside_corner(vec2 uv) {
    if (u_corner_radius <= 0.0) return false;
    vec2 p = uv * u_page_size;
    vec2 q = min(p, u_page_size - p);
    return q.x < u_corner_radius && q.y < u_corner_radius && length(vec2(u_corner_radius) - q) > u_corner_radius;
}
"""

SHADOW_VS = """
#version 330
uniform mat4 u_light_vp;
in vec3 in_pos;
in vec2 in_uv;
out vec3 v_world;
out vec2 v_uv;
void main() {
    v_world = in_pos;
    v_uv = in_uv;
    gl_Position = u_light_vp * vec4(in_pos, 1.0);
}
"""

# Stores the distance from the light, so shadow tests compare plain lengths.
# Shadow casters use an alpha mask; the page uses a blank (opaque) mask.
SHADOW_FS = """
#version 330
uniform vec3 u_light_pos;
uniform sampler2D u_mask;
in vec3 v_world;
in vec2 v_uv;
out float f_dist;
""" + _CORNER + """
void main() {
    if (texture(u_mask, v_uv).r < 0.5) discard;
    if (outside_corner(v_uv)) discard;
    f_dist = length(v_world - u_light_pos);
}
"""

SCENE_VS = """
#version 330
uniform mat4 u_view_proj;
uniform mat4 u_view;
in vec3 in_pos;
in vec2 in_uv;
in vec3 in_normal;
in float in_ao;
out vec3 v_world;
out vec2 v_uv;
// centroid: with MSAA, a pixel on a thin, edge-on triangle can be shaded at
// a point outside the triangle. Plain interpolation then extrapolates far
// past the vertex values; centroid keeps the sample inside.
centroid out vec3 v_normal;
centroid out float v_ao;
out float v_depth;
void main() {
    v_world = in_pos;
    v_uv = in_uv;
    v_normal = in_normal;
    v_ao = in_ao;
    v_depth = -(u_view * vec4(in_pos, 1.0)).z;
    gl_Position = u_view_proj * vec4(in_pos, 1.0);
}
"""

COLOR_FS = """
#version 330
const float PI = 3.14159265;

uniform vec3 u_camera_pos;
uniform sampler2D u_doc;

uniform sampler2D u_grain_tex;   // B channel: paper grain, zero mean
uniform vec2 u_page_size;        // cm
uniform float u_grain_tile;      // cm
uniform vec2 u_grain_offset;
uniform float u_grain;
uniform vec3 u_paper_tint;
uniform vec3 u_back_color;
uniform float u_roughness;
uniform float u_specular;
uniform float u_corner_radius;   // cm; > 0 = rounded card corners
uniform float u_glare;           // laminate reflection strength
uniform vec3 u_glare_dir;        // direction of the reflected ceiling/window light
uniform vec2 u_glare_cos;        // (outer, inner) cosines of its cone
uniform int u_glare_panel;       // 1 = a rectangular panel or window, 0 = a round cone
uniform vec3 u_glare_u;          // panel axes, perpendicular to u_glare_dir
uniform vec3 u_glare_v;
uniform vec2 u_glare_half;       // panel half sizes, as tangents of the angles
uniform float u_glare_edge;      // panel edge width, as a part of its half size
uniform float u_waviness;        // laminate surface wobble seen in reflections
uniform float u_room;            // faint reflection of the room on the laminate
uniform float u_wear;            // scratches and smudges that show in the reflections, 0..1
uniform float u_hologram;        // rainbow sheen strength

uniform sampler2D u_shadow;
uniform mat4 u_light_vp;
uniform float u_shadow_tan;      // tan(fov / 2) of the shadow frustum
uniform float u_shadow_texel;

uniform vec3 u_light_pos;
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_light_ref;
uniform float u_light_size;
uniform float u_ambient;
uniform vec3 u_ambient_color;
uniform vec3 u_ambient_dir;
uniform float u_ambient_focus;
uniform float u_wrap;
uniform bool u_display;          // true: gamma-encoded output for the editor's screen

in vec3 v_world;
in vec2 v_uv;
centroid in vec3 v_normal;
centroid in float v_ao;
in float v_depth;
out vec4 f_color;

const vec2 POISSON[16] = vec2[](
    vec2(-0.94201624, -0.39906216), vec2(0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870), vec2(0.34495938, 0.29387760),
    vec2(-0.91588581, 0.45771432), vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543, 0.27676845), vec2(0.97484398, 0.75648379),
    vec2(0.44323325, -0.97511554), vec2(0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023), vec2(0.79197514, 0.19090188),
    vec2(-0.24188840, 0.99706507), vec2(-0.81409955, 0.91437590),
    vec2(0.19984126, 0.78641367), vec2(0.14383161, -0.14100790)
);

// Percentage-closer soft shadows: find the average blocker distance, size
// the filter from the light size, then filter.
float shadow_factor(vec3 P, float ndl, float d_r) {
    vec4 lc = u_light_vp * vec4(P, 1.0);
    if (lc.w <= 0.0) return 1.0;
    vec2 suv = lc.xy / lc.w * 0.5 + 0.5;
    if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) return 1.0;

    float sin_t = sqrt(max(1.0 - ndl * ndl, 0.0));
    float bias = 0.02 + 0.05 * sin_t / max(ndl, 0.05);
    float uv_per_cm = 1.0 / (2.0 * d_r * u_shadow_tan);

    float a = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    mat2 rot = mat2(cos(a), sin(a), -sin(a), cos(a));

    float search = max(0.5 * u_light_size * uv_per_cm, 2.0 * u_shadow_texel);
    float blocker_sum = 0.0;
    int blockers = 0;
    for (int i = 0; i < 16; ++i) {
        float d = texture(u_shadow, suv + rot * POISSON[i] * search).r;
        if (d < d_r - bias) { blocker_sum += d; blockers++; }
    }
    if (blockers == 0) return 1.0;
    float d_b = blocker_sum / float(blockers);
    float penumbra_cm = u_light_size * (d_r - d_b) / d_b;
    float radius = max(0.5 * penumbra_cm * uv_per_cm, 1.0 * u_shadow_texel);

    float lit = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2 o = rot * POISSON[i] * radius;
        lit += texture(u_shadow, suv + o).r >= d_r - bias ? 1.0 : 0.0;
        lit += texture(u_shadow, suv - 0.5 * vec2(-o.y, o.x)).r >= d_r - bias ? 1.0 : 0.0;
    }
    return lit / 32.0;
}

float ggx_specular(vec3 N, vec3 L, vec3 V) {
    vec3 H = normalize(L + V);
    float nl = max(dot(N, L), 0.0), nv = max(dot(N, V), 1e-3), nh = max(dot(N, H), 0.0);
    float a = u_roughness * u_roughness;
    float a2 = a * a;
    float d = nh * nh * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * d * d);
    float k = a * 0.5;
    float G = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
    float F = u_specular + (1.0 - u_specular) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    return D * G * F / (4.0 * nv);  // already multiplied by n.l
}

// Laminate helpers (card look): value noise, the surface wobble and the room it mirrors.
// hash12 is "Hash without Sine" by David Hoskins, https://www.shadertoy.com/view/4djSRW,
// Copyright (c) 2014 David Hoskins, MIT License (full text in NOTICE).
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 w = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1.0, 0.0)), w.x),
               mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), w.x), w.y);
}

// A laminate is never quite flat: tilt the normal by a few degrees of smooth noise, in page centimeters.
vec3 wobble(vec3 N, vec2 uv) {
    if (u_waviness <= 0.0) return N;
    vec2 p = uv * u_page_size + u_grain_offset * 50.0;
    vec2 g = vec2(vnoise(p * 0.35) - 0.5, vnoise(p * 0.35 + 31.7) - 0.5)
           + 0.35 * vec2(vnoise(p * 0.9 + 7.1) - 0.5, vnoise(p * 0.9 + 53.2) - 0.5);
    return normalize(N + vec3(g * 0.02 * u_waviness, 0.0));
}

// Laminate wear in page centimeters: (fine scratches, smudges), each 0..1.
vec2 wear(vec2 p) {
    float scratch = 0.0;
    for (int k = 0; k < 3; k++) {
        float ang = 6.2831853 * hash12(u_grain_offset * 91.0 + float(k) * 7.3);
        vec2 q = mat2(cos(ang), -sin(ang), sin(ang), cos(ang)) * p * 45.0;  // 45 scratch rows per cm
        float row = floor(q.y);
        float on = step(0.95, hash12(vec2(row, float(k) * 13.1)))
                 * step(0.55, vnoise(vec2(q.x * 0.04 + row * 1.7, row)));  // sparse rows, broken into segments
        scratch = max(scratch, on * (1.0 - smoothstep(0.0, 0.12, abs(fract(q.y) - 0.5))));
    }
    float smudge = smoothstep(0.55, 0.8, vnoise(p * 0.9 + u_grain_offset * 23.0) * 0.7 + vnoise(p * 2.6) * 0.3);
    return vec2(scratch, smudge);
}

// How bright the room is in direction R: lighter overhead, darker low down, with large blotches for walls and
// furniture, and the photographer's dark shape where R points back at the camera.
float room(vec3 R, vec3 to_camera) {
    float b = 0.3 + 0.7 * clamp(R.z, 0.0, 1.0);
    vec2 q = R.xy / (abs(R.z) + 0.35) * 1.6 + u_grain_offset * 13.0;
    b *= 0.7 + 0.45 * vnoise(q) + 0.15 * vnoise(q * 2.7 + 5.0);
    float person = smoothstep(0.93, 0.985, dot(R, to_camera));
    return b * (1.0 - 0.5 * person);
}

void main() {
    if (u_corner_radius > 0.0) {  // card-look: rounded corners
        vec2 p = v_uv * u_page_size;
        vec2 q = min(p, u_page_size - p);
        if (q.x < u_corner_radius && q.y < u_corner_radius && length(vec2(u_corner_radius) - q) > u_corner_radius) discard;
    }
    // The crumple is in the geometry, so the vertex normals carry it.
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(u_camera_pos - v_world);

    vec2 st = v_uv * u_page_size / u_grain_tile + u_grain_offset;
    float grain = texture(u_grain_tex, st).b;
    vec3 paper = gl_FrontFacing ? texture(u_doc, v_uv).rgb * u_paper_tint : u_back_color;
    vec3 albedo = paper * (1.0 + u_grain * grain);
    float ao = clamp(v_ao, 0.0, 1.0);

    vec3 Lv = u_light_pos - v_world;
    float d = length(Lv);
    vec3 L = Lv / d;
    float ndl = dot(N, L);
    float diffuse = max((ndl + u_wrap) / (1.0 + u_wrap), 0.0);
    float atten = (u_light_ref * u_light_ref) / (d * d);
    float shadow = diffuse > 0.0 ? shadow_factor(v_world, clamp(ndl, 0.0, 1.0), d) : 0.0;
    vec3 light = u_light_color * (u_light_intensity * atten * shadow);

    // Room light: brighter on surfaces facing its main direction. Surfaces
    // facing away still get light bounced off the table and walls (the 0.3).
    float k = 1.0 + 6.0 * u_ambient_focus;
    float facing = 0.5 + 0.5 * dot(N, u_ambient_dir);
    float sky = 0.3 + 0.7 * pow(max(facing, 0.0), k);
    float sky_flat = 0.3 + 0.7 * pow(0.5 + 0.5 * u_ambient_dir.z, k);
    sky /= sky_flat;  // a flat, face-up page keeps the configured ambient level
    vec3 ambient = u_ambient * u_ambient_color * sky;

    vec3 direct = light * diffuse * mix(1.0, ao, 0.4);
    vec3 spec = light * ggx_specular(N, L, V);
    vec3 c = albedo * (direct + ambient * ao) + spec;
    if (gl_FrontFacing && (u_glare > 0.0 || u_hologram > 0.0 || u_room > 0.0)) {
        // Laminate: the mirror image of a broad light, Fresnel-weighted, the room around it, and a rainbow sheen.
        vec3 Nl = wobble(dot(N, N) > 1e-8 ? N : vec3(0.0, 0.0, 1.0), v_uv);
        vec3 Rf = reflect(-V, Nl);
        float fres = u_specular + (1.0 - u_specular) * pow(clamp(1.0 - dot(Nl, V), 0.0, 1.0), 5.0);
        float lobe, halo;
        if (u_glare_panel == 1) {
            // A ceiling panel or a window: a rectangle, brighter in its middle, with a faint uneven diffuser.
            // `halo` is a wider, soft version of it: where scratches and smudges catch the light.
            float dd = dot(Rf, u_glare_dir);
            vec2 ab = vec2(dot(Rf, u_glare_u), dot(Rf, u_glare_v)) / max(dd, 1e-3) / u_glare_half;
            float r = max(abs(ab.x), abs(ab.y));
            lobe = dd > 1e-3 ? (1.0 - smoothstep(1.0 - u_glare_edge, 1.0 + u_glare_edge, r))
                               * (0.75 + 0.25 * (1.0 - min(dot(ab, ab), 1.0))) * (0.85 + 0.3 * vnoise(ab * 2.5 + 4.0)) : 0.0;
            halo = dd > 1e-3 ? 1.0 - smoothstep(0.7, 1.6, r) : 0.0;
        } else {
            lobe = smoothstep(u_glare_cos.x, u_glare_cos.y, dot(Rf, u_glare_dir));
            halo = smoothstep(u_glare_cos.x - 0.15, u_glare_cos.y, dot(Rf, u_glare_dir));
        }
        if (u_wear > 0.0) {
            // Smudges dull the mirror and spread it into a haze; scratches scatter light where the glare is near.
            vec2 w = wear(v_uv * u_page_size) * u_wear;
            lobe = lobe * (1.0 - 0.5 * w.y) + 0.25 * w.y * halo;
            lobe += 1.5 * w.x * halo;
        }
        vec3 lam = u_glare * fres * lobe * u_ambient_color;
        if (u_room > 0.0) lam += 2.0 * u_room * fres * room(Rf, V) * u_ambient_color;
        if (u_hologram > 0.0) {
            float pat = 0.5 + 0.5 * sin(90.0 * v_uv.x + 35.0 * sin(9.0 * v_uv.y) + 20.0 * v_uv.y);
            float h = fract(2.5 * dot(Rf, vec3(0.8, 0.6, 0.0)) + 3.0 * v_uv.x + 1.7 * v_uv.y);
            vec3 rainbow = 0.5 + 0.5 * cos(6.2831853 * (h + vec3(0.0, 0.33, 0.67)));
            lam += u_hologram * (0.15 + 0.85 * lobe) * pat * rainbow * (0.3 + 0.7 * fres / max(u_specular, 1e-3) * 0.04);
        }
        if (!any(isnan(lam)) && !any(isinf(lam))) c += lam;
    }
    if (u_display) c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    f_color = vec4(c, 1.0);
}
"""

# (u, v, page mask, view depth). The mask is 1 only where the camera sees
# the printed (front) side of the page.
LABEL_FS = """
#version 330
uniform int u_kind;
in vec2 v_uv;
in float v_depth;
out vec4 f_label;
""" + _CORNER + """
void main() {
    if (u_kind == 1 && outside_corner(v_uv)) discard;
    if (u_kind == 1 && gl_FrontFacing) {
        f_label = vec4(v_uv, 1.0, v_depth);
    } else {
        f_label = vec4(0.0, 0.0, 0.0, v_depth);
    }
}
"""

# ---- table ------------------------------------------------------------------
# The table is the plane z = 0 under the page. It gets the same lamp, room
# light and shadow map as the page, so page and hand shadows fall on it.

TABLE_VS = """
#version 330
uniform mat4 u_view_proj;
uniform mat4 u_view;
in vec3 in_pos;
out vec3 v_world;
out float v_depth;
void main() {
    v_world = in_pos;
    v_depth = -(u_view * vec4(in_pos, 1.0)).z;
    gl_Position = u_view_proj * vec4(in_pos, 1.0);
}
"""

_TABLE_MAIN = """
uniform sampler2D u_height;      // page height above the table, seen from above
uniform vec4 u_height_rect;      // x, y, width, height (cm) that the height map covers
uniform float u_contact;         // contact shadow strength, 0..1
uniform float u_contact_radius;  // cm
uniform vec3 u_table_albedo;     // display mode only
uniform bool u_desk;             // a desk: output the light it reflects (light x desk color)
uniform bool u_desk_tex;         // the desk color comes from a texture
uniform sampler2D u_desk_map;
uniform vec2 u_desk_size;        // cm covered by one copy of the texture
uniform vec2 u_desk_offset;      // texture fractions
uniform float u_desk_rot;        // radians
uniform vec3 u_desk_color;

vec3 desk_albedo(vec2 p) {
    if (!u_desk_tex) return u_desk_color;
    float c = cos(u_desk_rot), s = sin(u_desk_rot);
    vec2 q = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
    return texture(u_desk_map, q / u_desk_size + u_desk_offset).rgb;
}

// Room light that the page blocks: high where the page is close above.
float contact_ao(vec2 p) {
    float occ = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2 st = (p + POISSON[i] * u_contact_radius - u_height_rect.xy) / u_height_rect.zw;
        float h = (any(lessThan(st, vec2(0.0))) || any(greaterThan(st, vec2(1.0)))) ? 1e3 : texture(u_height, st).r;
        occ += exp(-max(h, 0.0) / (0.5 * u_contact_radius));
    }
    return 1.0 - u_contact * occ / 16.0;
}

void main() {
    vec3 N = vec3(0.0, 0.0, 1.0);
    vec3 Lv = u_light_pos - v_world;
    float d = length(Lv);
    vec3 L = Lv / d;
    float ndl = dot(N, L);
    float diffuse = max((ndl + u_wrap) / (1.0 + u_wrap), 0.0);
    float atten = (u_light_ref * u_light_ref) / (d * d);
    float shadow = diffuse > 0.0 ? shadow_factor(v_world, clamp(ndl, 0.0, 1.0), d) : 0.0;
    float ao = contact_ao(v_world.xy);
    // The direct light is darkened a little too: right at the page edge the
    // shadow map cannot resolve the contact.
    vec3 direct = u_light_color * (u_light_intensity * atten * shadow * diffuse) * mix(1.0, ao, 0.5);
    vec3 ambient = u_ambient * u_ambient_color * ao;  // a face-up surface: the configured level
    vec3 E = direct + ambient;
    vec3 albedo = u_desk ? desk_albedo(v_world.xy) : u_table_albedo;
    if (u_display) {
        f_color = vec4(pow(clamp(E * albedo, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
    } else if (u_desk) {
        f_color = vec4(E * albedo, v_depth);  // light the desk reflects, and its camera depth
    } else {
        f_color = vec4(E, v_depth);  // light arriving at the table, and its camera depth
    }
}
"""

# The lighting code of COLOR_FS (uniforms, soft shadows) with the table's main.
TABLE_FS = COLOR_FS[:COLOR_FS.index("void main()")].replace(
    "in vec2 v_uv;\ncentroid in vec3 v_normal;\ncentroid in float v_ao;\n", "") + _TABLE_MAIN

# Page height seen from above, for the table's contact shadow. The depth test
# keeps the lowest page layer at each point.
HEIGHT_VS = """
#version 330
uniform mat4 u_proj;
in vec3 in_pos;
in vec2 in_uv;
out float v_height;
out vec2 v_uv;
void main() {
    v_height = in_pos.z;
    v_uv = in_uv;
    gl_Position = u_proj * vec4(in_pos, 1.0);
}
"""

HEIGHT_FS = """
#version 330
in float v_height;
in vec2 v_uv;
out float f_height;
""" + _CORNER + """
void main() {
    if (outside_corner(v_uv)) discard;
    f_height = v_height;
}
"""

# ---- desk items -------------------------------------------------------------
# Keyboards, mousepads, pens, other paper: plain meshes with a color per
# vertex (and an image on paper), lit like the page.

PROP_VS = """
#version 330
uniform mat4 u_view_proj;
uniform mat4 u_view;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;
out vec3 v_world;
out vec2 v_uv;
centroid out vec3 v_normal;
out vec3 v_color;
out float v_depth;
void main() {
    v_world = in_pos;
    v_uv = in_uv;
    v_normal = in_normal;
    v_color = in_color;
    v_depth = -(u_view * vec4(in_pos, 1.0)).z;
    gl_Position = u_view_proj * vec4(in_pos, 1.0);
}
"""

_PROP_MAIN = """
in vec3 v_color;
uniform bool u_has_image;
uniform sampler2D u_image;

void main() {
    vec3 N = normalize(v_normal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(u_camera_pos - v_world);
    vec3 albedo = v_color;
    if (u_has_image) albedo *= texture(u_image, v_uv).rgb * 0.95;
    vec3 Lv = u_light_pos - v_world;
    float d = length(Lv);
    vec3 L = Lv / d;
    float ndl = dot(N, L);
    float diffuse = max((ndl + u_wrap) / (1.0 + u_wrap), 0.0);
    float atten = (u_light_ref * u_light_ref) / (d * d);
    float shadow = diffuse > 0.0 ? shadow_factor(v_world, clamp(ndl, 0.0, 1.0), d) : 0.0;
    vec3 light = u_light_color * (u_light_intensity * atten * shadow);
    float k = 1.0 + 6.0 * u_ambient_focus;
    float sky = (0.3 + 0.7 * pow(max(0.5 + 0.5 * dot(N, u_ambient_dir), 0.0), k))
              / (0.3 + 0.7 * pow(0.5 + 0.5 * u_ambient_dir.z, k));
    vec3 c = albedo * (light * diffuse + u_ambient * u_ambient_color * sky) + light * ggx_specular(N, L, V);
    if (u_display) c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
    f_color = vec4(c, 1.0);
}
"""

PROP_FS = COLOR_FS[:COLOR_FS.index("void main()")].replace("centroid in float v_ao;\n", "") + _PROP_MAIN
