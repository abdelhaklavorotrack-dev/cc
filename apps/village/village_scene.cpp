// Procedural mountain village: terrain, houses, dry-stone walls, olive trees, props and colliders.

#include "village_scene.hpp"

#include "nova/core/log.hpp"
#include "nova/renderer/world_renderer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace nova::village {

using math::Mat4;
using math::Vec3;
using math::Vec4;
using renderer::Material;
using renderer::MeshData;

namespace {

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

f32 terrain_height_at(f32 x, f32 z) {
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
    std::vector<Collider> colliders;

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
        colliders.push_back({(a + b) * 0.5f + Vec3{0, height * 0.5f, 0}, {thickness * 0.5f, height * 0.5f + 0.1f, len * 0.5f}, yaw});
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
        colliders.push_back({h.position + Vec3{0, h.height * 0.5f, 0}, {h.width * 0.5f, h.height * 0.5f, h.depth * 0.5f}, h.yaw});
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
    b.colliders.push_back({tower + Vec3{0, 3.5f, 0}, {2.0f, 4.5f, 2.0f}, 0.0f});
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

} // namespace

f32 terrain_height(f32 x, f32 z) {
    return terrain_height_at(x, z);
}

Scene build_scene() {
    Builder b;
    build_village(b);
    Scene scene;
    scene.terrain = build_terrain();
    scene.opaque = std::move(b.opaque);
    scene.foliage = std::move(b.foliage);
    scene.two_sided = std::move(b.two_sided);
    scene.water = std::move(b.water);
    scene.colliders = std::move(b.colliders);
    return scene;
}

} // namespace nova::village
