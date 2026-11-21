#pragma once

#include "nova/math/math.hpp"

#include <vector>

namespace nova::renderer {

struct Vertex {
    math::Vec3 position;
    math::Vec3 normal;
};

// CPU-side mesh. Triangles are counter-clockwise when seen from outside.
struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<u32> indices;
};

// Axis-aligned cube centred at the origin with edge length `size` (24 vertices, flat normals).
[[nodiscard]] MeshData make_cube(f32 size = 1.0f);
// XZ plane centred at the origin facing +Y.
[[nodiscard]] MeshData make_plane(f32 size = 1.0f);

} // namespace nova::renderer
