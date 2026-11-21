#include "nova/renderer/mesh.hpp"

namespace nova::renderer {

using math::Vec3;

MeshData make_cube(f32 size) {
    const f32 h = size * 0.5f;
    struct Face {
        Vec3 normal, u, v; // u x v == normal, so (0,1,2),(0,2,3) is CCW from outside
    };
    const Face faces[6] = {
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},  {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},  {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},   {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},
    };
    MeshData mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);
    for (const Face& f : faces) {
        const u32 base = static_cast<u32>(mesh.vertices.size());
        const Vec3 c = f.normal * h;
        mesh.vertices.push_back({c - f.u * h - f.v * h, f.normal});
        mesh.vertices.push_back({c + f.u * h - f.v * h, f.normal});
        mesh.vertices.push_back({c + f.u * h + f.v * h, f.normal});
        mesh.vertices.push_back({c - f.u * h + f.v * h, f.normal});
        for (u32 i : {0u, 1u, 2u, 0u, 2u, 3u}) {
            mesh.indices.push_back(base + i);
        }
    }
    return mesh;
}

MeshData make_plane(f32 size) {
    const f32 h = size * 0.5f;
    const Vec3 n{0, 1, 0};
    MeshData mesh;
    mesh.vertices = {{{-h, 0, h}, n}, {{h, 0, h}, n}, {{h, 0, -h}, n}, {{-h, 0, -h}, n}};
    mesh.indices = {0, 1, 2, 0, 2, 3};
    return mesh;
}

} // namespace nova::renderer
