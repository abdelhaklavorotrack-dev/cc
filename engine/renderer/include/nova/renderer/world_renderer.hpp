#pragma once

#include "nova/math/math.hpp"
#include "nova/renderer/mesh.hpp"
#include "nova/renderer/renderer.hpp"
#include "nova/rhi/rhi.hpp"

#include <span>
#include <vector>

namespace nova::renderer {

// Material ids stored in Vertex::color.a; must match world.slang.
enum class Material : u8 {
    Plaster = 0,
    Stone = 1,
    Dirt = 2,
    Wood = 3,
    Foliage = 4,
    Water = 5,
    Terrain = 6, // shader blends dirt / dry grass / rock / snow by height, slope and noise
    Grass = 7,
    Emissive = 8,
    Flag = 9, // vertex colour rg = flag uv
    Cloth = 10,
    Metal = 11,
};

[[nodiscard]] constexpr math::Vec4 material_color(Material m, const math::Vec3& albedo) noexcept {
    return {albedo.x, albedo.y, albedo.z, static_cast<f32>(m)};
}

// Lighting and atmosphere for one view. Defaults give a late-afternoon sun.
struct WorldSettings {
    math::Vec3 sun_direction{0.8f, 0.27f, -0.36f}; // towards the sun
    math::Vec3 sun_color{1.0f, 0.74f, 0.50f};
    f32 sun_intensity = 4.6f;
    math::Vec3 sky_zenith{0.10f, 0.25f, 0.62f};
    math::Vec3 sky_horizon{0.56f, 0.68f, 0.86f};
    math::Vec3 ground_color{0.42f, 0.30f, 0.20f};
    f32 ambient = 0.7f;
    f32 fog_density = 0.0006f;
    f32 fog_height_falloff = 0.016f;
    f32 exposure = 0.75f;
    f32 time = 0.0f;
    // Area covered by the sun shadow map (orthographic, centred on shadow_center).
    math::Vec3 shadow_center{0, 0, 0};
    f32 shadow_radius = 50.0f;
};

struct WorldDrawItem {
    MeshId mesh;
    math::Mat4 transform = math::Mat4::identity();
    bool two_sided = false; // foliage, grass, flags
    bool casts_shadow = true;
};

struct WorldRenderStats {
    u32 draw_calls = 0;
    u32 triangles = 0;
};

// HDR forward renderer: sun shadow map -> sky + lit geometry into an RGBA16F target ->
// ACES tonemap and colour grade into the back buffer.
class WorldRenderer {
public:
    explicit WorldRenderer(rhi::Device& device, u32 shadow_map_size = 2048);
    ~WorldRenderer();
    WorldRenderer(const WorldRenderer&) = delete;
    WorldRenderer& operator=(const WorldRenderer&) = delete;

    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] MeshId upload(const MeshData& mesh);

    bool render_frame(const Camera& camera, const WorldSettings& settings, std::span<const WorldDrawItem> items);
    [[nodiscard]] const WorldRenderStats& stats() const noexcept { return stats_; }

private:
    struct GpuMesh {
        rhi::BufferHandle vertices;
        rhi::BufferHandle indices;
        u32 index_count = 0;
    };

    bool ensure_hdr_target(rhi::Extent extent);

    rhi::Device& device_;
    u32 shadow_size_;
    bool valid_ = false;
    rhi::ShaderHandle shaders_[8]{};
    rhi::PipelineHandle shadow_pipeline_;
    rhi::PipelineHandle sky_pipeline_;
    rhi::PipelineHandle main_pipeline_;
    rhi::PipelineHandle main_two_sided_pipeline_;
    rhi::PipelineHandle post_pipeline_;
    rhi::TextureHandle shadow_map_;
    rhi::TextureHandle hdr_target_;
    rhi::Extent hdr_extent_{};
    std::vector<rhi::BufferHandle> frame_buffers_;
    u32 frame_counter_ = 0;
    std::vector<GpuMesh> meshes_;
    WorldRenderStats stats_;
};

} // namespace nova::renderer
