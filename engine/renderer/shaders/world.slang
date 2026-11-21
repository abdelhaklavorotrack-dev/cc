// NOVA world shading: HDR forward pass with sun shadows, hemisphere ambient, procedural
// material detail, aerial perspective and an ACES/colour-graded post pass.
// FrameData layout must match WorldRenderer (renderer/src/world_renderer.cpp).

struct FrameData {
    float4x4 view_proj;
    float4x4 light_view_proj;
    float4 camera_pos;     // xyz, w = time (s)
    float4 camera_right;   // xyz, w = tan(fov/2) * aspect
    float4 camera_up;      // xyz, w = tan(fov/2)
    float4 camera_forward; // xyz, w = exposure
    float4 sun_dir;        // xyz towards the sun, w = shadow-map texel size
    float4 sun_color;      // rgb radiance
    float4 sky_zenith;     // rgb, w = fog density
    float4 sky_horizon;    // rgb, w = fog height falloff
    float4 ground_color;   // rgb, w = ambient strength
};

[[vk::binding(0, 0)]] ConstantBuffer<FrameData> frame;
[[vk::binding(1, 0)]] Sampler2D<float> shadow_map;
[[vk::binding(2, 0)]] Sampler2D<float4> hdr_scene;

struct ObjectData {
    float4x4 model;
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
        c = smoothstep(0.52, 0.8, c) * saturate(h * 6.0);
        float3 cloud = lerp(frame.sky_horizon.rgb * 1.1, frame.sun_color.rgb * 0.35 + 0.25, pow(sd, 3.0));
        col = lerp(col, cloud, c * 0.7);
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
    float3 fog_col = lerp(frame.sky_horizon.rgb, frame.sun_color.rgb * 0.45, pow(sd, 8.0) * 0.6);
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
};

[shader("vertex")]
float4 shadowVS(VSInput input) : SV_Position {
    float4 world = mul(object.model, float4(input.position, 1.0));
    return mul(frame.light_view_proj, world);
}

[shader("fragment")]
void shadowFS() {}

// ------------------------------------------------------------------ main pass

struct VSOutput {
    float4 position : SV_Position;
    float3 world_pos : POSITION0;
    float3 normal : NORMAL0;
    float4 color : COLOR0;
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

struct Surface {
    float3 albedo;
    float3 normal;
    float roughness;
    float3 emissive;
    float translucency;
    float wetness; // 1 = water
};

Surface terrain_surface(float3 p, float3 n) {
    Surface s;
    float r = length(p.xz);
    float slope = 1.0 - n.y;
    float large = fbm(p * 0.02, 4);
    float detail = fbm(p * 0.6, 4);

    float3 dirt = lerp(float3(0.36, 0.27, 0.19), float3(0.55, 0.44, 0.31), detail);
    dirt *= 0.8 + 0.35 * value_noise(p * 7.0); // pebbles and grit
    float3 dry_grass = lerp(float3(0.36, 0.33, 0.16), float3(0.52, 0.45, 0.24), detail);
    float3 rock = lerp(float3(0.30, 0.24, 0.19), float3(0.52, 0.44, 0.36), fbm(p * float3(0.15, 0.6, 0.15), 4));
    float3 snow = float3(0.90, 0.92, 0.96);

    float grass_w = smoothstep(0.45, 0.62, large) * (1.0 - smoothstep(0.35, 0.6, slope));
    grass_w *= smoothstep(18.0, 40.0, r); // keep the village streets bare
    float rock_w = smoothstep(0.28, 0.5, slope + (large - 0.5) * 0.3) * smoothstep(25.0, 60.0, r);
    float snow_w = smoothstep(150.0, 185.0, p.y + (large - 0.5) * 50.0) * (1.0 - smoothstep(0.5, 0.75, slope));

    float3 c = lerp(dirt, dry_grass, grass_w);
    c = lerp(c, rock, rock_w);
    c = lerp(c, snow, snow_w);
    s.albedo = c;
    s.normal = bump(n, p, rock_w > 0.5 ? 0.35 : 1.8, rock_w > 0.5 ? 0.6 : 0.25);
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
        float v = fbm(p * 3.0, 4);
        s.albedo = base * (0.75 + 0.5 * v);
        s.albedo = lerp(s.albedo, float3(0.30, 0.32, 0.22), smoothstep(0.62, 0.75, fbm(p * 1.3, 3)) * 0.5); // lichen
        s.normal = bump(n, p, 2.5, 0.7);
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
        s.normal = bump(n, p, 6.0, 1.2);
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

float ggx_specular(float3 n, float3 v, float3 l, float roughness) {
    float3 h = normalize(v + l);
    float a = max(roughness * roughness, 0.002);
    float a2 = a * a;
    float ndh = saturate(dot(n, h));
    float d = a2 / (3.14159 * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
    float ndv = saturate(dot(n, v)) + 1e-3;
    float ndl = saturate(dot(n, l));
    float k = a * 0.5;
    float vis = 1.0 / ((ndv * (1.0 - k) + k) * (ndl * (1.0 - k) + k));
    float fresnel = 0.04 + 0.96 * pow(1.0 - saturate(dot(h, v)), 5.0);
    return d * vis * fresnel * 0.25;
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
    n = s.normal;

    float3 v = normalize(frame.camera_pos.xyz - p);
    float3 l = frame.sun_dir.xyz;
    float ndl = saturate(dot(n, l));
    float shadow = sample_shadow(p, normalize(input.normal) * (front_face ? 1.0 : -1.0));

    // Contact occlusion: surfaces darken towards the ground they stand on.
    float ao = id == MAT_TERRAIN || id == MAT_WATER ? 1.0 : lerp(0.45, 1.0, saturate(p.y / 1.3));
    ao *= lerp(0.8, 1.0, saturate(n.y * 0.5 + 0.5));

    // Hemisphere ambient: sky from above, warm sunlit-ground bounce from below.
    float3 sky_amb = frame.sky_zenith.rgb * 0.9 + frame.sky_horizon.rgb * 0.3;
    float3 ground_amb = frame.ground_color.rgb * frame.sun_color.rgb * 0.25;
    float3 ambient = lerp(ground_amb, sky_amb, n.y * 0.5 + 0.5) * frame.ground_color.w;

    float3 direct = frame.sun_color.rgb * ndl * shadow;
    float3 color = s.albedo * (direct + ambient * ao);
    // Rough materials keep only a faint sheen; otherwise dusty ground glows at grazing angles.
    float spec_scale = lerp(1.0, 0.15, smoothstep(0.6, 0.95, s.roughness));
    color += frame.sun_color.rgb * ggx_specular(n, v, l, s.roughness) * ndl * shadow * spec_scale;

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
    color += s.emissive;

    color = apply_fog(color, p);
    return float4(color, 1.0);
}

// ------------------------------------------------------------------ post: tonemap + grade

float3 aces_fitted(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

[shader("vertex")]
FullscreenOut postVS(uint id : SV_VertexID) {
    return fullscreen(id);
}

[shader("fragment")]
float4 postFS(FullscreenOut input) : SV_Target {
    float3 hdr = hdr_scene.Sample(input.uv).rgb * frame.camera_forward.w;
    float3 c = aces_fitted(hdr);

    // Split toning: cool shadows, warm highlights (the "golden hour" look).
    float luma = dot(c, float3(0.2126, 0.7152, 0.0722));
    c *= lerp(float3(0.93, 0.98, 1.06), float3(1.05, 1.0, 0.93), smoothstep(0.1, 0.7, luma));
    // Gentle saturation and contrast.
    c = lerp(float3(luma, luma, luma), c, 1.12);
    c = saturate(c);
    c = c * c * (3.0 - 2.0 * c) * 0.25 + c * 0.75;

    // Vignette.
    float2 q = input.uv - 0.5;
    c *= 1.0 - dot(q, q) * 0.55;

    // Triangular dither hides banding in the 8-bit output.
    float noise = hash13(float3(input.position.xy, 1.0)) - hash13(float3(input.position.yx, 2.0));
    c += noise / 255.0;
    return float4(c, 1.0);
}
