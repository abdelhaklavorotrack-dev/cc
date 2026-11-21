#pragma once

// CPU-side 3D asset data (decoded, renderer-agnostic) and the glTF 2.0 importer.

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace nova::renderer {

// Tightly packed RGBA8 pixels, top row first.
struct ImageData {
    u32 width = 0;
    u32 height = 0;
    std::vector<u8> rgba;
};

enum class AlphaMode : u8 { Opaque, Mask, Blend };

// glTF metallic-roughness material. Image indices refer to ModelData::images; -1 = none.
struct MaterialData {
    math::Vec4 base_color_factor{1.0f, 1.0f, 1.0f, 1.0f};
    f32 metallic_factor = 1.0f;
    f32 roughness_factor = 1.0f;
    math::Vec3 emissive_factor{0.0f, 0.0f, 0.0f};
    f32 normal_scale = 1.0f;
    f32 occlusion_strength = 1.0f;
    AlphaMode alpha_mode = AlphaMode::Opaque;
    f32 alpha_cutoff = 0.5f;
    bool double_sided = false;
    i32 base_color_image = -1;         // sRGB
    i32 metallic_roughness_image = -1; // linear: G = roughness, B = metallic
    i32 normal_image = -1;             // linear, tangent space
    i32 occlusion_image = -1;          // linear: R
    i32 emissive_image = -1;           // sRGB
};

// One glTF primitive. Vertices are in mesh space; `transform` is the owning node's world matrix.
// Every vertex has a unit normal, a uv (0,0 if absent) and a tangent (generated when absent and
// the primitive has uvs). Vertex colour rgb = COLOR_0 (or white), a = 0.
struct ModelPrimitive {
    MeshData mesh;
    i32 material = -1; // index into ModelData::materials; -1 = default material
    math::Mat4 transform = math::Mat4::identity();
};

struct ModelData {
    std::vector<ImageData> images;
    std::vector<MaterialData> materials;
    std::vector<ModelPrimitive> primitives;
    // Axis-aligned bounds of all primitives after their node transforms.
    math::Vec3 bounds_min{0.0f, 0.0f, 0.0f};
    math::Vec3 bounds_max{0.0f, 0.0f, 0.0f};
};

// Loads a .gltf (external or data-URI buffers/images) or .glb file. Supports triangle primitives,
// indexed or not, any accessor component type, and PNG/JPEG images. On failure returns nullopt
// and, if `error` is non-null, a human-readable reason.
[[nodiscard]] std::optional<ModelData> load_gltf(const std::filesystem::path& path, std::string* error = nullptr);

} // namespace nova::renderer
