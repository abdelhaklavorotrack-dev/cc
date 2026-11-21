#pragma once

// The procedural mountain village shared by the village viewer and the Guerrilla demo game.

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"

#include <vector>

namespace nova::village {

// Solid obstacle: a box rotated about +Y. `center` is the middle of the box (not its base).
struct Collider {
    math::Vec3 center;
    math::Vec3 half_extents; // x across, y up, z along (before yaw)
    f32 yaw = 0.0f;
};

struct Scene {
    renderer::MeshData terrain;
    renderer::MeshData opaque;    // buildings, walls, props
    renderer::MeshData foliage;   // tree crowns
    renderer::MeshData two_sided; // grass, flag cloth
    renderer::MeshData water;     // puddles (no shadows)
    std::vector<Collider> colliders;
};

// Terrain height at world (x, z); the village street sits at y ~ 0.
[[nodiscard]] f32 terrain_height(f32 x, f32 z);

// Deterministic: the same scene every call.
[[nodiscard]] Scene build_scene();

} // namespace nova::village
