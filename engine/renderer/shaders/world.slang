// NOVA world shading: HDR forward pass with sun shadows, hemisphere ambient, procedural
// material detail, aerial perspective and an ACES/colour-graded post pass.
// FrameData layout must match WorldRenderer (renderer/src/world_renderer.cpp).

struct FrameData {
    float4x4 view_proj;
    float4x4 light_view_proj;
    float4x4 height_view_proj; // top-down orthographic view of the level (contact AO)
    float4 camera_pos;     // xyz, w = time (s)
    float4 camera_right;   // xyz, w = tan(fov/2) * aspect
    float4 camera_up;      // xyz, w = tan(fov/2)
    float4 camera_forward; // xyz, w = exposure
    float4 sun_dir;        // xyz towards the sun, w = shadow-map texel size
    float4 sun_color;      // rgb radiance
    float4 sky_zenith;     // rgb, w = fog density
    float4 sky_horizon;    // rgb, w = fog height falloff
    float4 ground_color;   // rgb, w = ambient strength
    float4 camera_clip;    // x near, y far, z contact-AO strength, w SSAO strength
    float4 height_params;  // x height-camera y, y its near plane, z depth range, w contact-AO radius
    float4 post_params;    // x bloom, y saturation, z colour bounce, w SSAO radius
    float4 screen;         // xy = 1 / size, zw = size (pixels) of the internal render targets
    float4 point_light;    // xyz position, w range (m)
    float4 point_color;    // rgb radiance (0 = off), w crosshair arm length (px, 0 = off)
    float4 output_size;    // xy back-buffer pixels
};

[[vk::binding(0, 0)]] ConstantBuffer<FrameData> frame;
[[vk::binding(1, 0)]] Sampler2D<float> shadow_map;
[[vk::binding(2, 0)]] Sampler2D<float4> hdr_scene;
[[vk::binding(3, 0)]] Sampler2D<float> scene_depth;
[[vk::binding(4, 0)]] Sampler2D<float4> bloom_tex;
[[vk::binding(5, 0)]] Sampler2D<float> height_map; // top-down depth of the level
[[vk::binding(6, 0)]] Sampler2D<float4> post_src;
// Asset (glTF metallic-roughness) material textures.
[[vk::binding(7, 0)]] Sampler2D<float4> base_color_tex;         // sRGB
[[vk::binding(8, 0)]] Sampler2D<float4> metallic_roughness_tex; // G roughness, B metallic
[[vk::binding(9, 0)]] Sampler2D<float4> normal_tex;             // tangent space
[[vk::binding(10, 0)]] Sampler2D<float4> occlusion_tex;         // R
[[vk::binding(11, 0)]] Sampler2D<float4> emissive_tex;          // sRGB

// Must match ObjectPush in world_renderer.cpp. Procedural draws push only `model`.
struct ObjectData {
    float4x4 model;
    float4 base_color; // factor (linear rgba)
    float4 material;   // x metallic, y roughness, z alpha cutoff (0 = opaque), w normal scale
    float4 emissive;   // rgb factor, w occlusion strength
};
[[vk::push_constant]] ConstantBuffer<ObjectData> object;

// Material ids (vertex colour alpha).
static const int MAT_PLASTER = 0;
static const int MAT_STONE = 1;
static const int MAT_DIRT = 2;
static const int MAT_WOOD = 3;
static const int MAT_FOLIAGE = 4;
static const int MAT_WATER = 5;
static const int MAT_TERRAIN = 6;
static const int MAT_GRASS = 7;
static const int MAT_EMISSIVE = 8;
static const int MAT_FLAG = 9;
static const int MAT_CLOTH = 10;
static const int MAT_METAL = 11;

// ------------------------------------------------------------------ noise

float hash13(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float value_noise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float n000 = hash13(i), n100 = hash13(i + float3(1, 0, 0));
    float n010 = hash13(i + float3(0, 1, 0)), n110 = hash13(i + float3(1, 1, 0));
    float n001 = hash13(i + float3(0, 0, 1)), n101 = hash13(i + float3(1, 0, 1));
    float n011 = hash13(i + float3(0, 1, 1)), n111 = hash13(i + float3(1, 1, 1));
    return lerp(lerp(lerp(n000, n100, u.x), lerp(n010, n110, u.x), u.y),
                lerp(lerp(n001, n101, u.x), lerp(n011, n111, u.x), u.y), u.z);
}

float fbm(float3 p, int octaves) {
    float sum = 0.0, amp = 0.5;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * value_noise(p);
        p = p * 2.03 + 17.1;
        amp *= 0.5;
    }
    return sum;
}

// Value noise with its analytic gradient in .yzw, so bump mapping needs one evaluation per octave.
float4 value_noise_d(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float3 du = 6.0 * f * (1.0 - f);
    float a = hash13(i), b = hash13(i + float3(1, 0, 0));
    float c = hash13(i + float3(0, 1, 0)), d = hash13(i + float3(1, 1, 0));
    float e = hash13(i + float3(0, 0, 1)), g = hash13(i + float3(1, 0, 1));
    float h = hash13(i + float3(0, 1, 1)), k = hash13(i + float3(1, 1, 1));
    float k1 = b - a, k2 = c - a, k3 = e - a;
    float k4 = a - b - c + d, k5 = a - c - e + h, k6 = a - b - e + g;
    float k7 = -a + b + c - d + e - g - h + k;
    float v = a + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x +
              k7 * u.x * u.y * u.z;
    float3 grad = du * float3(k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
                              k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
                              k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y);
    return float4(v, grad);
}

// Perturbs a normal with the gradient of a noise field (cheap procedural bump mapping).
float3 bump(float3 n, float3 p, float freq, float strength) {
    float3 g = 0.0;
    float amp = 0.5, scale = freq;
    float3 q = p * freq;
    for (int i = 0; i < 3; ++i) {
        g += amp * scale * value_noise_d(q).yzw;
        q = q * 2.03 + 17.1;
        scale *= 2.03;
        amp *= 0.5;
    }
    g -= n * dot(g, n); // tangential part only
    return normalize(n - g * strength);
}

// ------------------------------------------------------------------ sky and fog

float3 sky_color(float3 dir) {
    const float3 sun = frame.sun_dir.xyz;
    float h = dir.y;
    float3 col = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, pow(saturate(h), 0.5));
    // Below the horizon fade towards a hazy ground tone.
    col = lerp(col, frame.sky_horizon.rgb * 0.7 + frame.ground_color.rgb * 0.3, saturate(-h * 6.0));

    float sd = max(dot(dir, sun), 0.0);
    col += frame.sun_color.rgb * (pow(sd, 6.0) * 0.06 + pow(sd, 48.0) * 0.18);

    // Thin wispy clouds lit warm on the sun side.
    if (h > 0.0) {
        float2 uv = dir.xz / (h + 0.12) * 1.6;
        float c = fbm(float3(uv, frame.camera_pos.w * 0.01), 5);
        c = smoothstep(0.48, 0.72, c) * saturate(h * 6.0);
        float3 cloud = lerp(float3(0.92, 0.94, 1.0), frame.sun_color.rgb * 0.3 + 0.55, pow(sd, 3.0));
        col = lerp(col, cloud, c * 0.85);
    }

    // Sun disk.
    col += frame.sun_color.rgb * 12.0 * smoothstep(0.99955, 0.9998, sd);
    return col;
}

float3 apply_fog(float3 color, float3 world_pos) {
    float3 ro = frame.camera_pos.xyz;
    float3 v = world_pos - ro;
    float dist = length(v);
    float3 rd = v / max(dist, 1e-4);
    const float density = frame.sky_zenith.w;
    const float falloff = frame.sky_horizon.w;
    // Analytic integral of exponential height fog along the view ray.
    float ry = abs(rd.y) < 1e-3 ? 1e-3 : rd.y;
    float amount = density / falloff * exp(-ro.y * falloff) * (1.0 - exp(-dist * ry * falloff)) / ry;
    float f = saturate(1.0 - exp(-amount));
    float sd = max(dot(rd, frame.sun_dir.xyz), 0.0);
    float3 fog_col = lerp(frame.sky_horizon.rgb * 0.9, frame.sun_color.rgb * 0.3, pow(sd, 8.0) * 0.5);
    return lerp(color, fog_col, f);
}

// ------------------------------------------------------------------ fullscreen helpers

struct FullscreenOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

FullscreenOut fullscreen(uint id) {
    FullscreenOut o;
    o.uv = float2(float((id << 1) & 2), float(id & 2));
    o.position = float4(o.uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

[shader("vertex")]
FullscreenOut skyVS(uint id : SV_VertexID) {
    return fullscreen(id);
}

[shader("fragment")]
float4 skyFS(FullscreenOut input) : SV_Target {
    // Vulkan NDC y points down; uv.y = 0 is the top of the screen.
    float2 ndc = input.uv * 2.0 - 1.0;
    float3 dir = normalize(frame.camera_forward.xyz + frame.camera_right.xyz * ndc.x * frame.camera_right.w -
                           frame.camera_up.xyz * ndc.y * frame.camera_up.w);
    return float4(sky_color(dir), 1.0);
}

// ------------------------------------------------------------------ shadow pass

struct VSInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(1)]] float3 normal;
    [[vk::location(2)]] float4 color;
    [[vk::location(3)]] float2 uv;
    [[vk::location(4)]] float4 tangent;
};

[shader("vertex")]
float4 shadowVS(VSInput input) : SV_Position {
    float4 world = mul(object.model, float4(input.position, 1.0));
    return mul(frame.light_view_proj, world);
}

[shader("fragment")]
void shadowFS() {}

// Asset shadow casters: alpha-tested foliage, fences etc. cut holes in the shadow.
struct ShadowAssetOut {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
ShadowAssetOut shadowAssetVS(VSInput input) {
    ShadowAssetOut o;
    o.position = mul(frame.light_view_proj, mul(object.model, float4(input.position, 1.0)));
    o.uv = input.uv;
    return o;
}

[shader("fragment")]
void shadowAssetFS(ShadowAssetOut input) {
    if (object.material.z > 0.0 && base_color_tex.Sample(input.uv).a * object.base_color.a < object.material.z) {
        discard;
    }
}

// ------------------------------------------------------------------ main pass

struct VSOutput {
    float4 position : SV_Position;
    float3 world_pos : POSITION0;
    float3 normal : NORMAL0;
    float4 color : COLOR0;
    float2 uv : TEXCOORD0;
    float4 tangent : TANGENT0;
};

[shader("vertex")]
VSOutput mainVS(VSInput input) {
    VSOutput o;
    float4 world = mul(object.model, float4(input.position, 1.0));
    int id = int(input.color.a + 0.5);
    if (id == MAT_FOLIAGE || id == MAT_GRASS || id == MAT_FLAG) {
        // Wind sway, stronger higher up.
        float t = frame.camera_pos.w;
        float w = sin(t * 1.7 + world.x * 0.7 + world.z * 0.5) * 0.5 + sin(t * 3.1 + world.z * 1.3) * 0.25;
        float amp = id == MAT_FLAG ? 0.12 * input.color.r : (id == MAT_GRASS ? 0.04 : 0.025);
        world.xz += w * amp * float2(1.0, 0.6);
    }
    o.world_pos = world.xyz;
    o.position = mul(frame.view_proj, world);
    o.normal = mul((float3x3)object.model, input.normal);
    o.color = input.color;
    o.uv = input.uv;
    o.tangent = float4(mul((float3x3)object.model, input.tangent.xyz), input.tangent.w);
    return o;
}

static const float2 kPoisson[12] = {
    // The first four taps span the disk so they alone can classify fully lit/shadowed pixels.
    float2(-0.840, -0.074), float2(0.962, -0.195),  float2(0.519, 0.767),   float2(-0.322, -0.933),
    float2(-0.326, -0.406), float2(-0.696, 0.457),  float2(-0.203, 0.621),  float2(0.473, -0.480),
    float2(0.185, -0.893),  float2(0.507, 0.064),   float2(0.896, 0.412),   float2(-0.792, -0.598),
};

float sample_shadow(float3 p, float3 n) {
    const float texel = frame.sun_dir.w;
    float ndl = saturate(dot(n, frame.sun_dir.xyz));
    // Normal offset scaled by the light angle removes acne on grazing surfaces.
    float3 offset_p = p + n * (0.04 + 0.12 * (1.0 - ndl));
    float4 lp = mul(frame.light_view_proj, float4(offset_p, 1.0));
    float3 ndc = lp.xyz / lp.w;
    float2 uv = ndc.xy * 0.5 + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0) || ndc.z > 1.0) {
        return 1.0;
    }
    float lit = 0.0;
    // Rotate the disk per pixel to trade banding for fine noise.
    float angle = hash13(floor(p * 37.0)) * 6.2831;
    float2x2 rot = float2x2(cos(angle), -sin(angle), sin(angle), cos(angle));
    for (int i = 0; i < 4; ++i) {
        float2 o = mul(rot, kPoisson[i]) * texel * 1.8;
        lit += ndc.z - 0.0008 <= shadow_map.SampleLevel(uv + o, 0.0) ? 1.0 : 0.0;
    }
    if (lit == 0.0 || lit == 4.0) {
        return lit * 0.25; // not in a penumbra
    }
    for (int i = 4; i < 12; ++i) {
        float2 o = mul(rot, kPoisson[i]) * texel * 1.8;
        lit += ndc.z - 0.0008 <= shadow_map.SampleLevel(uv + o, 0.0) ? 1.0 : 0.0;
    }
    return lit / 12.0;
}

// Occlusion attenuates indirect light, not the hue of the receiving material.
float contact_ao(float3 p, float3 n) {
    const float strength = frame.camera_clip.z;
    const float R = frame.height_params.w;
    if (strength <= 0.0) {
        return 1.0;
    }
    float4 c0 = mul(frame.height_view_proj, float4(p, 1.0));
    float2 uv0 = c0.xy * 0.5 + 0.5;
    if (any(uv0 < 0.0) || any(uv0 > 1.0)) {
        return 1.0;
    }
    // Orthographic, so world xz maps to uv affinely.
    float2 du_dx = mul(frame.height_view_proj, float4(1, 0, 0, 0)).xy * 0.5;
    float2 du_dz = mul(frame.height_view_proj, float4(0, 0, 1, 0)).xy * 0.5;
    float2 centre = n.xz * R;
    float occ = 0.0, wsum = 0.0;
    for (int i = 0; i < 16; ++i) {
        float f = float(i) + 0.5;
        float a = f * 2.39996;
        float r = sqrt(f / 16.0) * R * 0.9;
        float2 q = centre + float2(cos(a), sin(a)) * r;
        float d = height_map.SampleLevel(uv0 + du_dx * q.x + du_dz * q.y, 0.0);
        float h = frame.height_params.x - (frame.height_params.y + d * frame.height_params.z);
        float w = 1.0 - r / R;
        occ += saturate((h - p.y - 0.2) / (R * 0.8)) * w;
        wsum += w;
    }
    return 1.0 - saturate(strength) * saturate(occ / wsum) * 0.65;
}

struct Surface {
    float3 albedo;
    float3 normal;
    float roughness;
    float3 emissive;
    float translucency;
    float wetness; // 1 = water
    float metallic;
    float occlusion; // material AO (1 = none)
    float env_spec;  // 1 = add sky-reflection ambient specular (asset materials)
};

Surface terrain_surface(float3 p, float3 n) {
    Surface s;
    float r = length(p.xz);
    float slope = 1.0 - n.y;
    float large = fbm(p * 0.02, 4);
    float detail = fbm(p * 0.6, 4);

    float3 dirt = lerp(float3(0.24, 0.16, 0.10), float3(0.43, 0.31, 0.20), detail);
    dirt *= 0.8 + 0.35 * value_noise(p * 7.0); // pebbles and grit
    float3 dry_grass = lerp(float3(0.20, 0.23, 0.10), float3(0.36, 0.34, 0.16), detail);
    float3 rock = lerp(float3(0.20, 0.17, 0.15), float3(0.42, 0.36, 0.31), fbm(p * float3(0.15, 0.6, 0.15), 4));
    float3 snow = float3(0.90, 0.92, 0.96);

    float grass_w = smoothstep(0.45, 0.62, large) * (1.0 - smoothstep(0.35, 0.6, slope));
    grass_w *= smoothstep(18.0, 40.0, r); // keep the village streets bare
    float rock_w = smoothstep(0.28, 0.5, slope + (large - 0.5) * 0.3) * smoothstep(25.0, 60.0, r);
    float snow_w = smoothstep(150.0, 185.0, p.y + (large - 0.5) * 50.0) * (1.0 - smoothstep(0.5, 0.75, slope));

    float3 c = lerp(dirt, dry_grass, grass_w);
    c = lerp(c, rock, rock_w);
    c = lerp(c, snow, snow_w);
    s.albedo = c;
    s.normal = bump(n, p, rock_w > 0.5 ? 0.35 : 1.8, rock_w > 0.5 ? 0.32 : 0.12);
    s.roughness = lerp(0.95, 0.6, snow_w);
    s.emissive = 0;
    s.translucency = 0;
    s.wetness = 0;
    return s;
}

Surface make_surface(int id, float3 base, float3 p, float3 n, float4 vcolor) {
    Surface s;
    s.albedo = base;
    s.normal = n;
    s.roughness = 0.85;
    s.emissive = 0;
    s.translucency = 0;
    s.wetness = 0;

    if (id == MAT_TERRAIN) {
        return terrain_surface(p, n);
    }
    if (id == MAT_PLASTER) {
        float blotch = fbm(p * 0.9, 4);
        float speck = value_noise(p * 25.0);
        float grime = saturate(1.0 - p.y / 1.4) * fbm(p * 2.5, 3); // splash dirt near the ground
        s.albedo = base * (0.82 + 0.3 * blotch) * (0.93 + 0.07 * speck);
        s.albedo = lerp(s.albedo, float3(0.36, 0.29, 0.22), grime * 0.7);
        // Peeled patches reveal the stone underneath.
        float peel = smoothstep(0.66, 0.7, fbm(p * 0.7 + 3.0, 4));
        s.albedo = lerp(s.albedo, float3(0.45, 0.39, 0.33), peel);
        s.normal = bump(n, p, 3.0, 0.12 + peel * 0.4);
    } else if (id == MAT_STONE) {
        float v = fbm(p * 10.0, 4);
        s.albedo = base * (0.50 + 0.80 * v);
        s.albedo = lerp(s.albedo, float3(0.30, 0.32, 0.22), smoothstep(0.62, 0.75, fbm(p * 1.3, 3)) * 0.5); // lichen
        s.normal = bump(n, p, 2.5, 0.16);
        s.roughness = 0.9;
    } else if (id == MAT_WOOD) {
        float grain = sin((p.y + p.x * 0.2 + fbm(p * float3(2, 18, 2), 3) * 0.8) * 55.0) * 0.5 + 0.5;
        s.albedo = base * (0.7 + 0.35 * grain) * (0.8 + 0.4 * fbm(p * 1.5, 3));
        s.normal = bump(n, p * float3(1, 6, 1), 4.0, 0.25);
        s.roughness = 0.8;
    } else if (id == MAT_FOLIAGE) {
        // High-frequency breakup reads as individual leaves and gaps at a distance.
        float leaves = value_noise(p * 9.0) * 0.6 + value_noise(p * 23.0) * 0.4;
        s.albedo = base * (0.55 + 0.6 * fbm(p * 2.2, 3)) * (0.35 + 0.9 * leaves);
        s.normal = bump(n, p, 6.0, 0.12);
        s.translucency = 0.45;
        s.roughness = 0.7;
    } else if (id == MAT_GRASS) {
        s.albedo = base * (0.75 + 0.5 * value_noise(p * 3.0));
        s.translucency = 0.35;
    } else if (id == MAT_WATER) {
        s.albedo = float3(0.04, 0.035, 0.03);
        s.normal = normalize(float3(sin(p.x * 4.0 + frame.camera_pos.w) * 0.01, 1.0,
                                    sin(p.z * 5.0 + frame.camera_pos.w * 1.3) * 0.01));
        s.roughness = 0.05;
        s.wetness = 1.0;
    } else if (id == MAT_EMISSIVE) {
        s.emissive = base * 6.0;
        s.albedo = base * 0.2;
    } else if (id == MAT_FLAG) {
        // vcolor.rg carries the flag's (u, v); Algerian flag: green | white with red crescent and star.
        float2 uv = vcolor.rg;
        float3 col = uv.x < 0.5 ? float3(0.0, 0.32, 0.12) : float3(0.92, 0.92, 0.9);
        float2 q = (uv - float2(0.5, 0.5)) * float2(1.5, 1.0);
        float outer = length(q - float2(0.0, 0.0));
        float inner = length(q - float2(0.07, 0.0));
        bool crescent = outer < 0.25 && inner > 0.2;
        float2 sq = q - float2(0.12, 0.0);
        float ang = atan2(sq.y, sq.x);
        float star_r = 0.09 * (0.55 + 0.45 * cos(5.0 * (ang + 3.14159)));
        bool star = length(sq) < star_r;
        if (crescent || star) {
            col = float3(0.75, 0.04, 0.06);
        }
        s.albedo = col * (0.85 + 0.15 * value_noise(float3(uv * 30.0, 0)));
        s.translucency = 0.3;
        s.roughness = 0.9;
    } else if (id == MAT_CLOTH) {
        float stripes = step(0.5, frac(p.y * 3.0 + fbm(p * 4.0, 2) * 0.3));
        float3 pattern = lerp(base, base * float3(0.5, 0.35, 0.3), stripes);
        s.albedo = pattern * (0.8 + 0.3 * fbm(p * 12.0, 2));
        s.normal = bump(n, p, 8.0, 0.1);
    } else if (id == MAT_METAL) {
        s.albedo = base * (0.6 + 0.4 * fbm(p * 6.0, 3));
        s.roughness = 0.45;
    } else if (id == MAT_DIRT) {
        s.albedo = base * (0.8 + 0.4 * fbm(p * 0.8, 4));
        s.normal = bump(n, p, 2.0, 0.3);
    }
    return s;
}

float3 ggx_specular(float3 n, float3 v, float3 l, float roughness, float3 f0) {
    float3 h = normalize(v + l);
    float a = max(roughness * roughness, 0.002);
    float a2 = a * a;
    float ndh = saturate(dot(n, h));
    float d = a2 / (3.14159 * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
    float ndv = saturate(dot(n, v)) + 1e-3;
    float ndl = saturate(dot(n, l));
    float k = a * 0.5;
    float vis = 1.0 / ((ndv * (1.0 - k) + k) * (ndl * (1.0 - k) + k));
    float3 fresnel = f0 + (1.0 - f0) * pow(1.0 - saturate(dot(h, v)), 5.0);
    return d * vis * fresnel * 0.25;
}

// Split-sum environment BRDF, analytic fit (Karis, "Physically Based Shading on Mobile").
float3 env_brdf(float3 f0, float roughness, float ndv) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// Both material paths use the geometric normal for sky fill and the shading normal for detail.
float3 light_surface(Surface s, float3 p, float3 geom_n) {
    float3 n = s.normal;
    float3 v = normalize(frame.camera_pos.xyz - p);
    float3 l = frame.sun_dir.xyz;
    float ndl = saturate(dot(n, l));
    float shadow = sample_shadow(p, geom_n);

    // Broad sky fill follows the surface, not each fine bump. Bounce includes skylight
    // even where the low sun contributes little to the ground.
    float3 sky_amb = lerp(frame.sky_horizon.rgb, frame.sky_zenith.rgb, 0.35);
    float3 ground_amb = frame.ground_color.rgb * (sky_amb + frame.sun_color.rgb * saturate(l.y));
    float3 ambient = lerp(ground_amb, sky_amb, geom_n.y * 0.5 + 0.5) * frame.ground_color.w;
    float ao = s.occlusion * contact_ao(p, geom_n);
    float3 diffuse_albedo = s.albedo * (1.0 - s.metallic);
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), s.albedo, s.metallic);
    float3 direct = frame.sun_color.rgb * ndl * shadow;
    float3 color = diffuse_albedo * (direct + ambient * ao);
    // Procedural rough materials keep only a faint sheen; otherwise dusty ground glows at grazing angles.
    float spec_scale = s.env_spec > 0.0 ? 1.0 : lerp(1.0, 0.15, smoothstep(0.6, 0.95, s.roughness));
    color += frame.sun_color.rgb * ggx_specular(n, v, l, s.roughness, f0) * ndl * shadow * spec_scale;

    if (s.env_spec > 0.0) {
        // Image-based-style ambient specular from the procedural sky (no cubemap needed): sharp sky
        // reflection for smooth surfaces, fading to the diffuse ambient as roughness grows.
        float ndv = saturate(dot(n, v)) + 1e-3;
        float3 r = reflect(-v, n);
        float3 env = lerp(sky_color(r) * frame.ground_color.w, ambient, saturate(s.roughness * 1.4));
        color += env * env_brdf(f0, s.roughness, ndv) * ao * s.env_spec;
    }
    if (s.translucency > 0.0) {
        float back = saturate(dot(-n, l)) * pow(saturate(dot(-v, l)), 2.0);
        color += s.albedo * frame.sun_color.rgb * back * s.translucency * shadow;
        color += s.albedo * frame.sun_color.rgb * 0.15 * s.translucency * shadow;
    }
    if (s.wetness > 0.0) {
        float3 r = reflect(-v, n);
        float fresnel = 0.02 + 0.98 * pow(1.0 - saturate(dot(n, v)), 5.0);
        color = lerp(color, sky_color(r) * lerp(0.35, 0.8, shadow), saturate(fresnel * 1.2 + 0.12));
    }
    // Point light (muzzle flash): inverse-square with a smooth range cut-off, unshadowed.
    if (any(frame.point_color.rgb > 0.0)) {
        float3 lv = frame.point_light.xyz - p;
        float d2 = dot(lv, lv);
        float3 pl = lv * rsqrt(max(d2, 1e-6));
        float window = saturate(1.0 - pow(d2 / (frame.point_light.w * frame.point_light.w), 2.0));
        float3 radiance = frame.point_color.rgb * window * window / (d2 + 0.04);
        float pndl = saturate(dot(n, pl));
        color += (diffuse_albedo + ggx_specular(n, v, pl, s.roughness, f0)) * radiance * pndl;
    }
    color += s.emissive;
    return apply_fog(color, p);
}

[shader("fragment")]
float4 mainFS(VSOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    int id = int(input.color.a + 0.5);
    float3 p = input.world_pos;
    float3 n = normalize(input.normal);
    if (!front_face) {
        n = -n; // two-sided foliage, grass and flags
    }
    Surface s = make_surface(id, input.color.rgb, p, n, input.color);
    s.metallic = 0.0;
    s.occlusion = 1.0;
    s.env_spec = 0.0;
    return float4(light_surface(s, p, n), 1.0);
}

// Emissive factors are relative to a sunlit white surface rather than absolute nits.
static const float kEmissiveScale = 4.0;

[shader("fragment")]
float4 assetFS(VSOutput input, bool front_face : SV_IsFrontFace) : SV_Target {
    float4 base = base_color_tex.Sample(input.uv) * object.base_color * float4(input.color.rgb, 1.0);
    if (object.material.z > 0.0 && base.a < object.material.z) {
        discard;
    }
    float3 p = input.world_pos;
    float3 geom_n = normalize(input.normal) * (front_face ? 1.0 : -1.0);
    float3 n = geom_n;
    float3 t = input.tangent.xyz;
    if (dot(t, t) > 1e-10) {
        t = normalize(t - geom_n * dot(geom_n, t));
        float3 b = cross(geom_n, t) * (input.tangent.w < 0.0 ? -1.0 : 1.0);
        float3 tn = normal_tex.Sample(input.uv).xyz * 2.0 - 1.0;
        tn.xy *= object.material.w;
        n = normalize(t * tn.x + b * tn.y + geom_n * max(tn.z, 1e-3));
    }
    float4 mr = metallic_roughness_tex.Sample(input.uv);

    Surface s;
    s.albedo = base.rgb;
    s.normal = n;
    s.metallic = saturate(object.material.x * mr.b);
    s.roughness = clamp(object.material.y * mr.g, 0.04, 1.0);
    s.occlusion = lerp(1.0, occlusion_tex.Sample(input.uv).r, object.emissive.w);
    s.emissive = emissive_tex.Sample(input.uv).rgb * object.emissive.rgb * kEmissiveScale;
    s.translucency = 0.0;
    s.wetness = 0.0;
    s.env_spec = 1.0;
    return float4(light_surface(s, p, geom_n), 1.0);
}

// ------------------------------------------------------------------ post: SSAO + colour bleed

float linear_depth(float d) {
    const float n = frame.camera_clip.x, f = frame.camera_clip.y;
    return n * f / (f - d * (f - n));
}

// World position of a screen uv at view depth t.
float3 screen_to_world(float2 uv, float t) {
    float2 ndc = uv * 2.0 - 1.0;
    float3 dir = frame.camera_forward.xyz + frame.camera_right.xyz * ndc.x * frame.camera_right.w -
                 frame.camera_up.xyz * ndc.y * frame.camera_up.w;
    return frame.camera_pos.xyz + dir * t;
}

float3 world_at(float2 uv) {
    return screen_to_world(uv, linear_depth(scene_depth.SampleLevel(uv, 0.0)));
}

float ssao(float2 uv, float3 p, float t, float2 pixel) {
    const float2 px = frame.screen.xy;
    // Normal from the neighbour pair closest in depth on each axis (stable across silhouettes).
    float3 l = world_at(uv - float2(px.x, 0)), r = world_at(uv + float2(px.x, 0));
    float3 u = world_at(uv - float2(0, px.y)), d = world_at(uv + float2(0, px.y));
    float3 dx = distance(r, p) < distance(l, p) ? r - p : p - l;
    float3 dy = distance(d, p) < distance(u, p) ? d - p : p - u;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, frame.camera_pos.xyz - p) < 0.0) {
        n = -n;
    }
    float3 t1 = normalize(abs(n.y) < 0.99 ? cross(n, float3(0, 1, 0)) : cross(n, float3(1, 0, 0)));
    float3 t2 = cross(n, t1);

    const float radius = frame.post_params.w * (1.0 + t * 0.02);
    float spin = hash13(float3(pixel, 3.0)) * 6.2831;
    float occ = 0.0;
    const int kSamples = 16;
    for (int i = 0; i < kSamples; ++i) {
        float f = (float(i) + 0.5) / float(kSamples);
        float a = float(i) * 2.39996 + spin;
        float z = sqrt(1.0 - f); // cosine-weighted hemisphere
        float rr = sqrt(f);
        float3 h = t1 * (cos(a) * rr) + t2 * (sin(a) * rr) + n * z;
        float scale = lerp(0.15, 1.0, f * f); // denser near the centre
        float3 s = p + n * 0.02 + h * radius * scale;
        float4 cs = mul(frame.view_proj, float4(s, 1.0));
        float2 suv = cs.xy / cs.w * 0.5 + 0.5;
        if (any(suv < 0.0) || any(suv > 1.0)) {
            continue;
        }
        float scene_t = linear_depth(scene_depth.SampleLevel(suv, 0.0));
        float range = saturate(radius / abs(t - scene_t));
        occ += (scene_t < cs.w - 0.03 ? 1.0 : 0.0) * range;
    }
    return 1.0 - occ / float(kSamples) * frame.camera_clip.w;
}

[shader("fragment")]
float4 compositeFS(FullscreenOut input) : SV_Target {
    const float2 uv = input.uv;
    const float exposure = frame.camera_forward.w;
    float3 c = hdr_scene.SampleLevel(uv, 0.0).rgb;
    float d = scene_depth.SampleLevel(uv, 0.0);
    // First-person viewmodel (< 0.6 m) is skipped: its scale is far below the SSAO radius.
    float view_t = d < 1.0 ? linear_depth(d) : 1e6;
    if (d < 1.0 && view_t > 0.6 && frame.camera_clip.w > 0.0) {
        c *= ssao(uv, screen_to_world(uv, view_t), view_t, input.position.xy);
    }

    // Screen-space colour bleed: nearby lit surfaces tint the shadows (cheap one-bounce look).
    const float bounce = frame.post_params.z;
    if (bounce > 0.0 && view_t > 0.6) {
        float3 e = c * exposure;
        float l = dot(e, float3(0.2126, 0.7152, 0.0722));
        float sh = 1.0 - smoothstep(0.05, 0.7, l);
        if (sh > 0.0) {
            const float2 step_px = frame.screen.xy * (90.0 * frame.screen.w / 900.0);
            float3 acc = 0.0;
            float wsum = 0.0;
            for (int i = 0; i < 16; ++i) {
                float f = float(i) + 0.5;
                float a = f * 2.39996;
                float2 o = float2(cos(a), sin(a)) * sqrt(f / 16.0) * step_px;
                float3 s = min(hdr_scene.SampleLevel(uv + o, 0.0).rgb * exposure, 4.0);
                float k = 1.0 + dot(s, 0.33);
                acc += s * k;
                wsum += k;
            }
            acc /= wsum;
            e += acc * bounce * sh * (e + 0.15) * 0.5;
            c = e / exposure;
        }
    }
    return float4(c, 1.0);
}

// ------------------------------------------------------------------ post: bloom (quarter resolution)

[shader("fragment")]
float4 bloomDownFS(FullscreenOut input) : SV_Target {
    const float2 px = frame.screen.xy; // full-resolution texel
    float3 c = post_src.SampleLevel(input.uv + float2(-px.x, -px.y), 0.0).rgb;
    c += post_src.SampleLevel(input.uv + float2(px.x, -px.y), 0.0).rgb;
    c += post_src.SampleLevel(input.uv + float2(-px.x, px.y), 0.0).rgb;
    c += post_src.SampleLevel(input.uv + float2(px.x, px.y), 0.0).rgb;
    c *= 0.25 * frame.camera_forward.w;
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    // Soft-knee threshold around 0.9 (exposed, pre-tonemap units).
    float knee = clamp(l - 0.6, 0.0, 0.6);
    float w = max(knee * knee / 2.4, l - 0.9) / max(l, 1e-4);
    return float4(min(c * w, 16.0), 1.0);
}

float3 blur9(float2 uv, float2 dir) {
    const float w[5] = {0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};
    float3 c = post_src.SampleLevel(uv, 0.0).rgb * w[0];
    for (int i = 1; i < 5; ++i) {
        c += post_src.SampleLevel(uv + dir * float(i), 0.0).rgb * w[i];
        c += post_src.SampleLevel(uv - dir * float(i), 0.0).rgb * w[i];
    }
    return c;
}

[shader("fragment")]
float4 bloomBlurHFS(FullscreenOut input) : SV_Target {
    return float4(blur9(input.uv, float2(frame.screen.x * 6.0, 0.0)), 1.0);
}

[shader("fragment")]
float4 bloomBlurVFS(FullscreenOut input) : SV_Target {
    return float4(blur9(input.uv, float2(0.0, frame.screen.y * 6.0)), 1.0);
}

// ------------------------------------------------------------------ post: tonemap, AA, grade

float3 aces_fitted(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 tonemapped(float2 uv) {
    float3 hdr = hdr_scene.SampleLevel(uv, 0.0).rgb * frame.camera_forward.w;
    hdr += bloom_tex.SampleLevel(uv, 0.0).rgb * frame.post_params.x;
    return aces_fitted(hdr);
}

float luma_of(float3 c) {
    return dot(c, float3(0.299, 0.587, 0.114));
}

[shader("vertex")]
FullscreenOut postVS(uint id : SV_VertexID) {
    return fullscreen(id);
}

[shader("fragment")]
float4 postFS(FullscreenOut input) : SV_Target {
    const float2 uv = input.uv;
    const float2 px = frame.screen.xy;

    // FXAA (console variant) on the tonemapped image.
    float3 m = tonemapped(uv);
    float l_nw = luma_of(tonemapped(uv + float2(-px.x, -px.y)));
    float l_ne = luma_of(tonemapped(uv + float2(px.x, -px.y)));
    float l_sw = luma_of(tonemapped(uv + float2(-px.x, px.y)));
    float l_se = luma_of(tonemapped(uv + float2(px.x, px.y)));
    float l_m = luma_of(m);
    float l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    float l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    float3 c = m;
    if (l_max - l_min > max(0.0312, l_max * 0.125)) {
        float2 dir = float2(-((l_nw + l_ne) - (l_sw + l_se)), (l_nw + l_sw) - (l_ne + l_se));
        float reduce = max((l_nw + l_ne + l_sw + l_se) * 0.03125, 1.0 / 128.0);
        float inv = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
        dir = clamp(dir * inv, -8.0, 8.0) * px;
        float3 a = 0.5 * (tonemapped(uv + dir * (1.0 / 3.0 - 0.5)) + tonemapped(uv + dir * (2.0 / 3.0 - 0.5)));
        float3 b = a * 0.5 + 0.25 * (tonemapped(uv - dir * 0.5) + tonemapped(uv + dir * 0.5));
        float l_b = luma_of(b);
        c = (l_b < l_min || l_b > l_max) ? a : b;
    }

    // The sun supplies the warmth; keep neutral shadows neutral after tonemapping.
    c *= 1.0 - 0.10 * smoothstep(0.35, 0.71, distance(uv, 0.5));
    float l = dot(c, float3(0.2126, 0.7152, 0.0722));
    c = saturate(lerp(l.xxx, c, frame.post_params.y));

    // Crosshair: four arms around the centre with a dark outline.
    const float arm = frame.point_color.w;
    if (arm > 0.0) {
        float2 q = abs(input.position.xy - frame.output_size.xy * 0.5);
        const float gap = arm * 0.6;
        float along = max(q.x, q.y);
        float across = min(q.x, q.y);
        bool in_arm = along >= gap && along <= gap + arm;
        if (in_arm && across <= 1.0) {
            c = lerp(c, float3(0.95, 0.95, 0.9), 0.85);
        } else if (along >= gap - 1.0 && along <= gap + arm + 1.0 && across <= 2.0) {
            c *= 0.35;
        }
    }

    // Triangular dither hides banding in the 8-bit output.
    float noise = hash13(float3(input.position.xy, 1.0)) - hash13(float3(input.position.yx, 2.0));
    c += noise / 255.0;
    return float4(c, 1.0);
}
