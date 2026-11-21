// NOVA village demo: a procedurally built mountain village at golden hour, rendered with the
// WorldRenderer (sun shadows, HDR, aerial perspective, procedural materials, ACES + grading).
//
//   nova_village [--headless] [--frames N] [--width W --height H] [--no-vsync] [--validation]
//                [--screenshot out.ppm]
//
// Window controls: hold right mouse to look, WASD move, Q/E down/up, Shift faster, Esc quit.
// Exit code 77 means no usable Vulkan device.

#include "nova/core/log.hpp"
#include "nova/core/time.hpp"
#include "nova/platform/window.hpp"
#include "nova/renderer/world_renderer.hpp"
#include "nova/rhi/rhi.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

using namespace nova;
using math::Mat4;
using math::Vec3;
using math::Vec4;
using renderer::Material;
using renderer::MeshData;

namespace {

constexpr int kExitNoGpu = 77;

// ------------------------------------------------------------------ deterministic randomness / noise

struct Rng {
    u32 state;
    explicit Rng(u32 seed) : state(seed * 747796405u + 2891336453u) {}
    u32 next() {
        state = state * 747796405u + 2891336453u;
        u32 w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (w >> 22u) ^ w;
    }
    f32 uniform() { return static_cast<f32>(next() >> 8) / 16777216.0f; }
    f32 range(f32 lo, f32 hi) { return lo + (hi - lo) * uniform(); }
};

f32 hash2(i32 x, i32 y) {
    u32 h = static_cast<u32>(x) * 374761393u + static_cast<u32>(y) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return static_cast<f32>((h ^ (h >> 16)) & 0xFFFFFF) / 16777215.0f;
}

f32 value_noise(f32 x, f32 y) {
    const f32 fx = std::floor(x), fy = std::floor(y);
    const i32 ix = static_cast<i32>(fx), iy = static_cast<i32>(fy);
    const f32 tx = x - fx, ty = y - fy;
    const f32 ux = tx * tx * (3 - 2 * tx), uy = ty * ty * (3 - 2 * ty);
    const f32 a = hash2(ix, iy), b = hash2(ix + 1, iy), c = hash2(ix, iy + 1), d = hash2(ix + 1, iy + 1);
    return math::lerp(math::lerp(a, b, ux), math::lerp(c, d, ux), uy);
}

f32 fbm(f32 x, f32 y, int octaves) {
    f32 sum = 0, amp = 0.5f;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * value_noise(x, y);
        x = x * 2.03f + 13.7f;
        y = y * 2.03f + 7.3f;
        amp *= 0.5f;
    }
    return sum;
}

f32 ridged(f32 x, f32 y, int octaves) {
    f32 sum = 0, amp = 0.5f, weight = 1.0f;
    for (int i = 0; i < octaves; ++i) {
        f32 n = 1.0f - std::abs(value_noise(x, y) * 2.0f - 1.0f);
        n *= n * weight;
        weight = std::clamp(n * 1.6f, 0.0f, 1.0f);
        sum += n * amp;
        x = x * 2.1f + 5.1f;
        y = y * 2.1f + 1.7f;
        amp *= 0.5f;
    }
    return sum;
}

f32 smoothstep(f32 e0, f32 e1, f32 x) {
    const f32 t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
}

// ------------------------------------------------------------------ terrain

constexpr Vec3 kTowerHill{-30.0f, 0.0f, -150.0f};

f32 terrain_height(f32 x, f32 z) {
    const f32 r = std::sqrt(x * x + z * z);
    const f32 hills = (fbm(x * 0.012f, z * 0.012f, 4) - 0.35f) * 22.0f * smoothstep(32.0f, 95.0f, r);
    const f32 mountains = ridged(x * 0.0035f + 3.1f, z * 0.0035f, 6) * 230.0f * smoothstep(140.0f, 420.0f, r);
    const f32 dx = x - kTowerHill.x, dz = z - kTowerHill.z;
    const f32 hill = 26.0f * std::exp(-(dx * dx + dz * dz) / (2.0f * 38.0f * 38.0f));
    const f32 ground = (fbm(x * 0.3f, z * 0.3f, 2) - 0.5f) * 0.06f; // keeps the streets almost flat
    return std::max(hills, 0.0f) + mountains + hill + ground;
}

MeshData build_terrain() {
    constexpr int kCells = 260;
    constexpr f32 kSize = 1300.0f;
    constexpr f32 kStep = kSize / kCells;
    const Vec3 origin{-kSize * 0.5f, 0.0f, -kSize * 0.72f}; // extends further towards the mountains
    MeshData mesh;
    mesh.vertices.reserve((kCells + 1) * (kCells + 1));
    for (int j = 0; j <= kCells; ++j) {
        for (int i = 0; i <= kCells; ++i) {
            const f32 x = origin.x + static_cast<f32>(i) * kStep, z = origin.z + static_cast<f32>(j) * kStep;
            const f32 e = kStep * 0.5f;
            const Vec3 n = math::normalize(Vec3{terrain_height(x - e, z) - terrain_height(x + e, z), 2.0f * e,
                                                terrain_height(x, z - e) - terrain_height(x, z + e)});
            mesh.vertices.push_back(
                {{x, terrain_height(x, z), z}, n, renderer::material_color(Material::Terrain, {1, 1, 1})});
        }
    }
    for (int j = 0; j < kCells; ++j) {
        for (int i = 0; i < kCells; ++i) {
            const u32 a = j * (kCells + 1) + i, b = a + 1, c = a + (kCells + 1), d = c + 1;
            mesh.indices.insert(mesh.indices.end(), {a, c, d, a, d, b}); // CCW seen from above
        }
    }
    return mesh;
}

// ------------------------------------------------------------------ village builder

class Builder {
public:
    MeshData opaque;     // buildings, walls, props
    MeshData foliage;    // tree crowns
    MeshData two_sided;  // grass, flag cloth
    MeshData water;      // puddles (no shadows)

    Builder() : cube_(renderer::make_cube(1.0f)), sphere_(renderer::make_icosphere(1)),
                cylinder_(renderer::make_cylinder(10, 1, 1, 1)) {}

    static Mat4 place(const Vec3& pos, const Vec3& size, f32 yaw = 0, f32 tilt = 0) {
        return math::translate(pos) * math::rotate({0, 1, 0}, yaw) * math::rotate({1, 0, 0}, tilt) * math::scale(size);
    }

    void box(const Vec3& center, const Vec3& size, Material m, const Vec3& albedo, f32 yaw = 0) {
        renderer::append(opaque, cube_, place(center, size, yaw), renderer::material_color(m, albedo));
    }

    // Box in a house's local frame (x across the facade, +z out of the front face).
    void local_box(const Mat4& frame, const Vec3& center, const Vec3& size, Material m, const Vec3& albedo) {
        renderer::append(opaque, cube_, frame * math::translate(center) * math::scale(size),
                         renderer::material_color(m, albedo));
    }

    // Irregular angular stone: a box whose eight corners are jittered, with flat-shaded faces.
    void stone(MeshData& dst, const Vec3& center, const Vec3& radii, f32 yaw, const Vec3& albedo, Material m) {
        std::array<Vec3, 8> corner_offset;
        for (Vec3& o : corner_offset) {
            o = {rng_.range(-0.5f, 0.5f), rng_.range(-0.45f, 0.45f), rng_.range(-0.5f, 0.5f)};
        }
        MeshData s = cube_;
        for (auto& v : s.vertices) {
            const int idx = (v.position.x > 0 ? 1 : 0) | (v.position.y > 0 ? 2 : 0) | (v.position.z > 0 ? 4 : 0);
            // Pull corners inwards so stones read as rounded-off slabs rather than boxes.
            v.position = v.position * 1.4f + corner_offset[static_cast<usize>(idx)] * 0.8f;
        }
        for (usize f = 0; f < s.indices.size(); f += 6) {
            auto& v0 = s.vertices[s.indices[f]];
            const Vec3 n = math::normalize(math::cross(s.vertices[s.indices[f + 1]].position - v0.position,
                                                       s.vertices[s.indices[f + 2]].position - v0.position));
            for (usize k = 0; k < 6; ++k) {
                s.vertices[s.indices[f + k]].normal = n;
            }
        }
        renderer::append(dst, s, place(center, radii, yaw), renderer::material_color(m, albedo));
    }

    void dry_stone_wall(Vec3 a, Vec3 b, f32 height, f32 thickness) {
        const Vec3 d = b - a;
        const f32 len = math::length(d);
        const f32 yaw = std::atan2(d.x, d.z);
        const Vec3 dir = d * (1.0f / len);
        // Core fills the gaps between stones.
        box((a + b) * 0.5f + Vec3{0, height * 0.45f, 0}, {thickness * 0.8f, height * 0.9f, len}, Material::Stone,
            {0.30f, 0.27f, 0.23f}, yaw);
        for (f32 y = 0.1f; y < height;) {
            const f32 row_h = rng_.range(0.17f, 0.3f);
            for (f32 t = rng_.range(0.0f, 0.3f); t < len;) {
                const f32 sl = rng_.range(0.25f, 0.55f);
                const f32 tone = rng_.range(0.7f, 1.2f);
                const Vec3 c = a + dir * (t + sl * 0.5f) + Vec3{0, y + row_h * 0.5f, 0};
                for (int side = -1; side <= 1; side += 2) {
                    const Vec3 sideways{dir.z * static_cast<f32>(side), 0, -dir.x * static_cast<f32>(side)};
                    stone(opaque, c + sideways * (thickness * 0.33f),
                          {thickness * rng_.range(0.14f, 0.22f), row_h * 0.58f, sl * 0.56f},
                          yaw + rng_.range(-0.3f, 0.3f), Vec3{0.42f, 0.36f, 0.29f} * tone, Material::Stone);
                }
                t += sl;
            }
            y += row_h;
        }
        // Capstones.
        for (f32 t = 0.2f; t < len; t += rng_.range(0.45f, 0.65f)) {
            stone(opaque, a + dir * t + Vec3{0, height + 0.08f, 0}, {thickness * 0.5f, 0.16f, 0.3f},
                  yaw + rng_.range(-0.4f, 0.4f), Vec3{0.6f, 0.55f, 0.47f} * rng_.range(0.8f, 1.1f), Material::Stone);
        }
        grass_along(a, b, 1.0f);
    }

    struct HouseSpec {
        Vec3 position; // centre of the footprint at ground level
        f32 width;     // along the facade
        f32 depth;
        f32 height;
        f32 yaw;       // 0 => facade faces +Z
        Vec3 plaster{0.78f, 0.68f, 0.55f};
        f32 door_offset = 0.0f;
        bool lantern = false;
    };

    Mat4 house(const HouseSpec& h) {
        const Mat4 frame = math::translate(h.position) * math::rotate({0, 1, 0}, h.yaw);
        const Vec3 wood{0.36f, 0.25f, 0.16f};
        local_box(frame, {0, h.height * 0.5f, 0}, {h.width, h.height, h.depth}, Material::Plaster, h.plaster);
        // Stone plinth.
        local_box(frame, {0, 0.25f, 0}, {h.width + 0.12f, 0.5f, h.depth + 0.12f}, Material::Stone,
                  {0.5f, 0.45f, 0.38f});
        // Flat earthen roof with a lip and protruding beam ends.
        local_box(frame, {0, h.height + 0.12f, 0}, {h.width + 0.5f, 0.24f, h.depth + 0.5f}, Material::Dirt,
                  {0.55f, 0.45f, 0.34f});
        for (f32 x = -h.width * 0.5f + 0.35f; x < h.width * 0.5f; x += 0.65f) {
            local_box(frame, {x, h.height - 0.12f, h.depth * 0.5f + 0.2f}, {0.14f, 0.14f, 0.5f}, Material::Wood, wood);
        }
        // Door with lintel.
        const f32 door_h = std::min(2.3f, h.height - 0.6f);
        local_box(frame, {h.door_offset, door_h * 0.5f, h.depth * 0.5f + 0.02f}, {1.15f, door_h, 0.1f},
                  Material::Wood, {0.32f, 0.22f, 0.14f});
        local_box(frame, {h.door_offset, door_h + 0.1f, h.depth * 0.5f + 0.06f}, {1.6f, 0.18f, 0.2f}, Material::Wood,
                  wood);
        // Small dark windows.
        for (f32 x : {-h.width * 0.3f, h.width * 0.3f}) {
            if (std::abs(x - h.door_offset) > 1.4f) {
                local_box(frame, {x, std::min(2.6f, h.height - 1.0f), h.depth * 0.5f + 0.01f}, {0.6f, 0.7f, 0.06f},
                          Material::Wood, {0.08f, 0.06f, 0.05f});
            }
        }
        if (h.lantern) {
            const Vec3 l{h.door_offset + 1.3f, 2.5f, h.depth * 0.5f};
            local_box(frame, l + Vec3{0, 0.25f, 0.25f}, {0.05f, 0.05f, 0.5f}, Material::Metal, {0.15f, 0.13f, 0.1f});
            local_box(frame, l + Vec3{0, 0.0f, 0.45f}, {0.22f, 0.3f, 0.22f}, Material::Emissive, {1.0f, 0.62f, 0.3f});
            local_box(frame, l + Vec3{0, 0.18f, 0.45f}, {0.26f, 0.06f, 0.26f}, Material::Metal, {0.12f, 0.1f, 0.08f});
            local_box(frame, l + Vec3{0, -0.18f, 0.45f}, {0.26f, 0.06f, 0.26f}, Material::Metal, {0.12f, 0.1f, 0.08f});
        }
        grass_along(math::transform_point(frame, {-h.width * 0.5f, 0, h.depth * 0.5f + 0.15f}),
                    math::transform_point(frame, {h.width * 0.5f, 0, h.depth * 0.5f + 0.15f}), 0.6f);
        return frame;
    }

    void olive_tree(const Vec3& base, const Vec3& crown_center, f32 crown_radius) {
        const Vec3 bark{0.28f, 0.22f, 0.17f};
        // Gnarled trunk: bent segments from the base to the crown.
        Vec3 p = base;
        const int segments = 6;
        for (int i = 0; i < segments; ++i) {
            const f32 t = static_cast<f32>(i + 1) / segments;
            Vec3 q = math::lerp(base, crown_center - Vec3{0, crown_radius * 0.4f, 0}, t);
            q += Vec3{rng_.range(-0.25f, 0.25f), 0, rng_.range(-0.25f, 0.25f)} * (1.0f - t);
            segment(p, q, 0.28f * (1.0f - t * 0.55f), bark);
            p = q;
        }
        // Branches.
        for (int i = 0; i < 5; ++i) {
            const Vec3 tip = crown_center + Vec3{rng_.range(-1, 1), rng_.range(-0.3f, 0.6f), rng_.range(-1, 1)} *
                                                crown_radius * 0.8f;
            segment(p, tip, 0.09f, bark);
        }
        // Crown: many small jittered clumps of leaves.
        const int blobs = static_cast<int>(48 * crown_radius);
        for (int i = 0; i < blobs; ++i) {
            Vec3 o{rng_.range(-1, 1), rng_.range(-0.45f, 0.55f), rng_.range(-1, 1)};
            if (math::length(o) > 1.0f) {
                o = math::normalize(o);
            }
            const f32 r = crown_radius * rng_.range(0.12f, 0.24f);
            MeshData blob = sphere_;
            for (auto& v : blob.vertices) {
                v.position = v.position * (0.6f + 0.7f * rng_.uniform());
            }
            const f32 tone = rng_.range(0.8f, 1.2f);
            renderer::append(foliage, blob, place(crown_center + o * crown_radius, {r * 1.2f, r * 0.8f, r * 1.2f},
                                                  rng_.range(0, 6.28f)),
                             renderer::material_color(Material::Foliage, Vec3{0.25f, 0.29f, 0.13f} * tone));
        }
    }

    void segment(const Vec3& a, const Vec3& b, f32 radius, const Vec3& albedo) {
        const Vec3 d = b - a;
        const f32 len = math::length(d);
        const Vec3 dir = d * (1.0f / len);
        const Vec3 axis = math::cross({0, 1, 0}, dir);
        const f32 angle = std::acos(std::clamp(dir.y, -1.0f, 1.0f));
        const Mat4 rot = math::length(axis) > 1e-4f ? math::rotate(axis, angle) : Mat4::identity();
        renderer::append(opaque, cylinder_, math::translate(a) * rot * math::scale({radius, len * 1.05f, radius}),
                         renderer::material_color(Material::Wood, albedo));
    }

    void grass_along(const Vec3& a, const Vec3& b, f32 density) {
        const f32 len = math::length(b - a);
        const int count = static_cast<int>(len * 2.2f * density);
        for (int i = 0; i < count; ++i) {
            if (rng_.uniform() < 0.35f) {
                continue;
            }
            const Vec3 p = math::lerp(a, b, rng_.uniform()) + Vec3{rng_.range(-0.3f, 0.3f), 0, rng_.range(-0.3f, 0.3f)};
            grass_tuft(p, rng_.range(0.25f, 0.6f));
        }
    }

    void grass_tuft(const Vec3& p, f32 height) {
        const Vec3 color = Vec3{0.50f, 0.46f, 0.22f} * rng_.range(0.75f, 1.15f);
        for (int i = 0; i < 6; ++i) {
            const f32 a = rng_.range(0, 6.28f);
            const Vec3 lean{std::cos(a) * rng_.range(0.05f, 0.25f), 0, std::sin(a) * rng_.range(0.05f, 0.25f)};
            const Vec3 side{-std::sin(a) * 0.025f, 0, std::cos(a) * 0.025f};
            const Vec3 tip = p + lean + Vec3{0, height * rng_.range(0.6f, 1.0f), 0};
            const Vec3 n = math::normalize(math::cross(side, tip - p));
            const u32 base = static_cast<u32>(two_sided.vertices.size());
            const Vec4 c = renderer::material_color(Material::Grass, color);
            two_sided.vertices.push_back({p - side, n, c});
            two_sided.vertices.push_back({p + side, n, c});
            two_sided.vertices.push_back({tip, n, c});
            two_sided.indices.insert(two_sided.indices.end(), {base, base + 1, base + 2});
        }
    }

    void flag(const Vec3& pole_top, f32 width, f32 height) {
        constexpr int kSeg = 12;
        const u32 base = static_cast<u32>(two_sided.vertices.size());
        for (int j = 0; j <= 1; ++j) {
            for (int i = 0; i <= kSeg; ++i) {
                const f32 u = static_cast<f32>(i) / kSeg;
                const f32 wave = std::sin(u * 5.0f) * 0.12f * u; // resting ripple; the shader adds wind
                // rg = flag uv (u also scales wind sway in the vertex shader).
                two_sided.vertices.push_back({pole_top + Vec3{u * width, -static_cast<f32>(j) * height, wave},
                                              {0, 0, 1},
                                              {u, static_cast<f32>(j), 0.0f, static_cast<f32>(Material::Flag)}});
            }
        }
        for (int i = 0; i < kSeg; ++i) {
            const u32 a = base + i, b = a + 1, c = a + kSeg + 1, d = c + 1;
            two_sided.indices.insert(two_sided.indices.end(), {a, c, d, a, d, b});
        }
    }

    void puddle(const Vec3& center, f32 size) {
        for (int i = 0; i < 3; ++i) {
            const Vec3 o{rng_.range(-0.5f, 0.5f) * size, 0, rng_.range(-0.4f, 0.4f) * size};
            renderer::append(water, sphere_,
                             place(center + o + Vec3{0, 0.025f, 0},
                                   {size * rng_.range(0.4f, 0.7f), 0.012f, size * rng_.range(0.3f, 0.5f)},
                                   rng_.range(0, 3.14f)),
                             renderer::material_color(Material::Water, {0.05f, 0.05f, 0.05f}));
        }
    }

    void scatter_rocks(const Vec3& a, const Vec3& b, int count) {
        for (int i = 0; i < count; ++i) {
            const Vec3 p = math::lerp(a, b, rng_.uniform()) + Vec3{rng_.range(-0.6f, 0.6f), 0, rng_.range(-0.6f, 0.6f)};
            const f32 s = rng_.range(0.08f, 0.35f);
            stone(opaque, p + Vec3{0, s * 0.3f, 0}, {s, s * 0.6f, s * 1.2f}, rng_.range(0, 6.28f),
                  Vec3{0.58f, 0.52f, 0.44f} * rng_.range(0.8f, 1.1f), Material::Stone);
        }
    }

    MeshData& cylinder_target() { return opaque; }
    const MeshData& cylinder() const { return cylinder_; }
    Rng& rng() { return rng_; }

private:
    MeshData cube_;
    MeshData sphere_;
    MeshData cylinder_;
    Rng rng_{1954};
};

struct Village {
    MeshData terrain;
    Builder b;
};

void build_village(Builder& b) {
    constexpr f32 kHalfPi = math::kPi * 0.5f;
    const Vec3 warm_plaster{0.72f, 0.60f, 0.46f};
    const Vec3 pale_plaster{0.76f, 0.68f, 0.57f};

    // Left side of the street (facades face +X).
    b.dry_stone_wall({-3.0f, 0, 11.0f}, {-3.0f, 0, 2.6f}, 1.7f, 0.75f);
    b.box({-2.58f, 1.15f, 8.2f}, {0.05f, 1.15f, 0.95f}, Material::Cloth, {0.45f, 0.10f, 0.07f}); // rug on the wall
    b.house({{-6.2f, 0, -3.6f}, 12.0f, 6.0f, 5.2f, kHalfPi, warm_plaster, 1.6f, true});
    b.house({{-6.6f, 0, -16.5f}, 9.0f, 6.0f, 4.2f, kHalfPi, pale_plaster, -1.0f});
    b.house({{-7.0f, 0, -27.5f}, 8.0f, 7.0f, 3.8f, kHalfPi, warm_plaster, 0.5f});
    b.dry_stone_wall({-3.2f, 0, -9.7f}, {-3.2f, 0, -12.0f}, 1.2f, 0.6f);

    // Right side (facades face -X).
    b.dry_stone_wall({3.0f, 0, 11.0f}, {3.0f, 0, -3.6f}, 1.35f, 0.7f);
    const Mat4 house_c = b.house({{7.6f, 0, -9.0f}, 9.0f, 7.0f, 3.8f, -kHalfPi, pale_plaster, 2.0f});
    b.house({{8.2f, 0, -22.0f}, 8.0f, 7.5f, 4.4f, -kHalfPi, warm_plaster, -1.2f, true});
    b.house({{9.0f, 0, -33.0f}, 7.0f, 7.0f, 3.6f, -kHalfPi, pale_plaster});

    // Ladder leaning on house C.
    for (f32 side : {-0.25f, 0.25f}) {
        b.box({3.75f, 1.6f, -6.0f + side}, {0.07f, 3.4f, 0.07f}, Material::Wood, {0.4f, 0.3f, 0.2f});
    }
    for (f32 y = 0.4f; y < 3.2f; y += 0.4f) {
        b.box({3.75f, y, -6.0f}, {0.05f, 0.05f, 0.5f}, Material::Wood, {0.38f, 0.28f, 0.18f});
    }
    // Woven baskets.
    for (const Vec3& p : {Vec3{3.7f, 0, -3.2f}, Vec3{3.6f, 0, -2.4f}}) {
        renderer::append(b.cylinder_target(), b.cylinder(), math::translate(p) * math::scale({0.38f, 0.55f, 0.38f}),
                         renderer::material_color(Material::Wood, {0.55f, 0.43f, 0.26f}));
    }
    // Flag pole on the roof of house C.
    const Vec3 pole_base = math::transform_point(house_c, {2.5f, 3.9f, -1.0f});
    renderer::append(b.cylinder_target(), b.cylinder(),
                     math::translate(pole_base) * math::scale({0.05f, 6.0f, 0.05f}),
                     renderer::material_color(Material::Metal, {0.35f, 0.33f, 0.3f}));
    b.flag(pole_base + Vec3{0.05f, 5.9f, 0}, 2.1f, 1.4f);

    // End of the street (facade faces +Z, towards the camera).
    b.house({{0.0f, 0, -40.0f}, 10.0f, 6.0f, 4.6f, 0.0f, warm_plaster, 0.0f, true});
    b.house({{-12.0f, 0, -42.0f}, 8.0f, 6.0f, 5.0f, 0.3f, pale_plaster});
    b.house({{13.0f, 0, -44.0f}, 9.0f, 6.0f, 4.0f, -0.2f, warm_plaster});
    // Market awning.
    b.box({3.6f, 2.5f, -29.0f}, {2.6f, 0.04f, 3.0f}, Material::Cloth, {0.82f, 0.76f, 0.62f});

    // Olive trees.
    b.olive_tree({-4.4f, 0, 6.0f}, {-2.6f, 4.6f, 5.2f}, 2.4f);
    b.olive_tree({-3.9f, 0, -11.0f}, {-3.4f, 4.0f, -11.2f}, 1.9f);
    b.olive_tree({11.0f, 0, -14.0f}, {11.0f, 4.4f, -14.0f}, 2.2f);
    b.olive_tree({-15.0f, 0, -20.0f}, {-15.0f, 4.8f, -20.0f}, 2.6f);

    // Street dressing.
    b.puddle({0.6f, 0, -5.5f}, 1.7f);
    b.puddle({-0.8f, 0, -12.5f}, 2.4f);
    b.puddle({1.3f, 0, -19.5f}, 1.2f);
    b.scatter_rocks({-2.3f, 0, 9.0f}, {-2.3f, 0, -20.0f}, 26);
    b.scatter_rocks({2.3f, 0, 9.0f}, {2.3f, 0, -20.0f}, 22);
    b.grass_along({-2.2f, 0, 10.0f}, {-2.2f, 0, -25.0f}, 0.8f);
    b.grass_along({2.4f, 0, 10.0f}, {2.4f, 0, -25.0f}, 0.8f);

    // Watchtower on the hill and scattered houses on the slopes.
    const Vec3 tower{kTowerHill.x, terrain_height(kTowerHill.x, kTowerHill.z), kTowerHill.z};
    renderer::append(b.cylinder_target(), b.cylinder(),
                     math::translate(tower - Vec3{0, 1, 0}) * math::scale({2.2f, 9.0f, 2.2f}),
                     renderer::material_color(Material::Stone, {0.62f, 0.55f, 0.45f}));
    renderer::append(b.cylinder_target(), renderer::make_cylinder(10, 3.0f, 0.2f, 3.0f),
                     math::translate(tower + Vec3{0, 8.0f, 0}),
                     renderer::material_color(Material::Wood, {0.35f, 0.24f, 0.16f}));
    Rng& rng = b.rng();
    for (int i = 0; i < 26; ++i) {
        const f32 a = rng.range(-2.6f, -0.5f); // mostly in front of the camera
        const f32 r = rng.range(48.0f, 120.0f);
        const f32 x = std::cos(a) * r, z = std::sin(a) * r;
        const f32 w = rng.range(5.0f, 9.0f);
        b.house({{x, terrain_height(x, z) - 0.4f, z}, w, rng.range(5.0f, 7.0f), rng.range(3.2f, 5.0f),
                 rng.range(0, 6.28f), rng.uniform() < 0.5f ? warm_plaster : pale_plaster, 0.0f});
        if (rng.uniform() < 0.6f) {
            const f32 tx = x + rng.range(-8, 8), tz = z + rng.range(-8, 8);
            const f32 ty = terrain_height(tx, tz);
            b.olive_tree({tx, ty, tz}, {tx, ty + 3.8f, tz}, rng.range(1.6f, 2.4f));
        }
    }
}

template <typename T>
void parse_number(const char* s, T& out) {
    std::from_chars(s, s + std::strlen(s), out);
}

bool write_ppm(const std::string& path, const std::vector<u8>& rgba, u32 w, u32 h) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    for (usize i = 0; i < rgba.size(); i += 4) {
        std::fwrite(&rgba[i], 1, 3, f);
    }
    return std::fclose(f) == 0;
}

} // namespace

int main(int argc, char** argv) {
    platform::WindowDesc wdesc;
    wdesc.title = "NOVA Village";
    wdesc.width = 1600;
    wdesc.height = 900;
    rhi::DeviceDesc ddesc;
    u64 max_frames = 0;
    std::string screenshot;
#if defined(NOVA_ENABLE_ASSERTS)
    ddesc.validation = true;
#endif
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--headless") {
            wdesc.backend = platform::Backend::Headless;
        } else if (arg == "--frames" && has_value) {
            parse_number(argv[++i], max_frames);
        } else if (arg == "--width" && has_value) {
            parse_number(argv[++i], wdesc.width);
        } else if (arg == "--height" && has_value) {
            parse_number(argv[++i], wdesc.height);
        } else if (arg == "--no-vsync") {
            ddesc.vsync = false;
        } else if (arg == "--validation") {
            ddesc.validation = true;
        } else if (arg == "--screenshot" && has_value) {
            screenshot = argv[++i];
        }
    }
    const bool headless = wdesc.backend == platform::Backend::Headless;
    if (headless && max_frames == 0) {
        max_frames = 60;
    }

    auto window = platform::create_window(wdesc);
    if (!window) {
        NOVA_FATAL("Village", "could not create a window");
        return 1;
    }
    ddesc.window = headless ? nullptr : window.get();
    ddesc.width = wdesc.width;
    ddesc.height = wdesc.height;
    auto device = rhi::Device::create(ddesc);
    if (!device) {
        NOVA_ERROR("Village", "no usable Vulkan device");
        return kExitNoGpu;
    }
    renderer::WorldRenderer world(*device);
    if (!world.valid()) {
        return 1;
    }

    Stopwatch build_timer;
    Builder builder;
    build_village(builder);
    const MeshData terrain = build_terrain();
    std::vector<renderer::WorldDrawItem> items = {
        {world.upload(terrain)},
        {world.upload(builder.opaque)},
        {world.upload(builder.foliage)},
        {world.upload(builder.two_sided), Mat4::identity(), true},
        {world.upload(builder.water), Mat4::identity(), false, false},
    };
    const usize tris = (terrain.indices.size() + builder.opaque.indices.size() + builder.foliage.indices.size() +
                        builder.two_sided.indices.size() + builder.water.indices.size()) / 3;
    NOVA_INFO("Village", "built scene in {:.0f} ms: {} triangles in {} draws", build_timer.elapsed_ms(), tris,
              items.size());

    renderer::Camera camera;
    camera.position = {1.1f, 1.7f, 8.5f};
    camera.fov_y = math::radians(62.0f);
    camera.z_far = 2000.0f;
    f32 yaw = math::radians(-93.5f);  // looking down the street (-Z)
    f32 pitch = math::radians(1.5f);
    renderer::WorldSettings settings;
    settings.shadow_center = {0, 0, -12};
    settings.shadow_radius = 42.0f;

    platform::Input input;
    FrameStats cpu_stats;
    FrameStats gpu_stats;
    Stopwatch frame_timer;
    Stopwatch report_timer;
    f32 time = 3.0f;

    for (u64 frame = 0; max_frames == 0 || frame < max_frames; ++frame) {
        const f64 frame_ms = frame_timer.lap_ms();
        if (frame > 0) {
            cpu_stats.record(frame_ms);
        }
        const f32 dt = headless ? 1.0f / 60.0f : static_cast<f32>(std::min(frame_ms, 100.0) / 1000.0);
        time += dt;

        input.begin_frame();
        if (!window->poll_events(input) || input.was_pressed(platform::Key::Escape)) {
            break;
        }
        if (input.is_down(platform::MouseButton::Right)) {
            yaw += input.mouse_dx() * 0.003f;
            pitch = std::clamp(pitch - input.mouse_dy() * 0.003f, -1.4f, 1.4f);
        }
        const Vec3 forward{std::cos(pitch) * std::cos(yaw), std::sin(pitch), std::cos(pitch) * std::sin(yaw)};
        const Vec3 right = math::normalize(math::cross(forward, {0, 1, 0}));
        const f32 speed = (input.is_down(platform::Key::LeftShift) ? 20.0f : 5.0f) * dt;
        if (input.is_down(platform::Key::W)) camera.position += forward * speed;
        if (input.is_down(platform::Key::S)) camera.position -= forward * speed;
        if (input.is_down(platform::Key::D)) camera.position += right * speed;
        if (input.is_down(platform::Key::A)) camera.position -= right * speed;
        if (input.is_down(platform::Key::E)) camera.position.y += speed;
        if (input.is_down(platform::Key::Q)) camera.position.y -= speed;
        camera.target = camera.position + forward;

        const platform::WindowSize size = window->size();
        if (!headless && (size.width != device->back_buffer_extent().width ||
                          size.height != device->back_buffer_extent().height)) {
            device->resize(size.width, size.height);
        }

        settings.time = time;
        world.render_frame(camera, settings, items);
        if (device->last_gpu_frame_ms() >= 0.0) {
            gpu_stats.record(device->last_gpu_frame_ms());
        }
        if (report_timer.elapsed_ms() >= 1000.0) {
            report_timer.reset();
            const auto c = cpu_stats.summarize();
            const auto g = gpu_stats.summarize();
            NOVA_INFO("Village", "CPU frame {:.2f} ms (p99 {:.2f}) | GPU {:.2f} ms (p99 {:.2f}) | {:.0f} fps",
                      c.avg_ms, c.p99_ms, g.avg_ms, g.p99_ms, c.avg_fps);
        }
    }

    device->wait_idle();
    int exit_code = 0;
    if (!screenshot.empty()) {
        std::vector<u8> pixels;
        const rhi::Extent e = device->back_buffer_extent();
        if (headless && device->read_back(pixels) && write_ppm(screenshot, pixels, e.width, e.height)) {
            NOVA_INFO("Village", "wrote {}", screenshot);
        } else {
            NOVA_ERROR("Village", "failed to write screenshot {} (requires --headless)", screenshot);
            exit_code = 1;
        }
    }
    const auto c = cpu_stats.summarize();
    const auto g = gpu_stats.summarize();
    NOVA_INFO("Village", "exit: CPU avg {:.2f} ms p99 {:.2f} ms | GPU avg {:.2f} ms p99 {:.2f} ms | {} draws, {} tris",
              c.avg_ms, c.p99_ms, g.avg_ms, g.p99_ms, world.stats().draw_calls, world.stats().triangles);
    if (device->validation_error_count() > 0) {
        NOVA_ERROR("Village", "{} Vulkan validation errors", device->validation_error_count());
        exit_code = 1;
    }
    return exit_code;
}
