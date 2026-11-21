#include "nova/renderer/world_renderer.hpp"

#include "nova/core/log.hpp"

#include "world_spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace nova::renderer {
namespace {

using math::Mat4;
using math::Vec3;

// Must match FrameData in world.slang (std140: matrices then float4s).
struct FrameData {
    Mat4 view_proj;
    Mat4 light_view_proj;
    f32 camera_pos[4];
    f32 camera_right[4];
    f32 camera_up[4];
    f32 camera_forward[4];
    f32 sun_dir[4];
    f32 sun_color[4];
    f32 sky_zenith[4];
    f32 sky_horizon[4];
    f32 ground_color[4];
};
static_assert(sizeof(FrameData) == 128 + 9 * 16);

void set4(f32 (&dst)[4], const Vec3& v, f32 w) {
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = w;
}

enum ShaderSlot { kSkyVS, kSkyFS, kShadowVS, kShadowFS, kMainVS, kMainFS, kPostVS, kPostFS };

constexpr rhi::Format kHdrFormat = rhi::Format::RGBA16_Float;

} // namespace

WorldRenderer::WorldRenderer(rhi::Device& device, u32 shadow_map_size)
    : device_(device), shadow_size_(shadow_map_size) {
    struct Entry {
        const char* name;
        rhi::ShaderStage stage;
    };
    const Entry entries[8] = {
        {"skyVS", rhi::ShaderStage::Vertex},    {"skyFS", rhi::ShaderStage::Fragment},
        {"shadowVS", rhi::ShaderStage::Vertex}, {"shadowFS", rhi::ShaderStage::Fragment},
        {"mainVS", rhi::ShaderStage::Vertex},   {"mainFS", rhi::ShaderStage::Fragment},
        {"postVS", rhi::ShaderStage::Vertex},   {"postFS", rhi::ShaderStage::Fragment},
    };
    for (int i = 0; i < 8; ++i) {
        shaders_[i] = device_.create_shader({world_spv, entries[i].name, entries[i].stage});
        if (!shaders_[i].valid()) {
            NOVA_ERROR("Renderer", "failed to create world shader {}", entries[i].name);
            return;
        }
    }

    const rhi::BindingDesc frame_binding{0, rhi::BindingType::UniformBuffer};
    const std::vector<rhi::VertexAttribute> attributes = {
        {0, rhi::VertexFormat::Float3, static_cast<u32>(offsetof(Vertex, position))},
        {1, rhi::VertexFormat::Float3, static_cast<u32>(offsetof(Vertex, normal))},
        {2, rhi::VertexFormat::Float4, static_cast<u32>(offsetof(Vertex, color))},
    };

    rhi::GraphicsPipelineDesc shadow;
    shadow.vertex_shader = shaders_[kShadowVS];
    shadow.fragment_shader = shaders_[kShadowFS];
    shadow.vertex_stride = sizeof(Vertex);
    shadow.attributes = attributes;
    shadow.push_constant_size = sizeof(Mat4);
    shadow.bindings = {frame_binding};
    shadow.cull = rhi::CullMode::None; // thin walls and foliage must cast from both sides
    shadow.depth_bias = true;
    shadow.color_format = rhi::Format::Undefined;
    shadow_pipeline_ = device_.create_graphics_pipeline(shadow);

    rhi::GraphicsPipelineDesc sky;
    sky.vertex_shader = shaders_[kSkyVS];
    sky.fragment_shader = shaders_[kSkyFS];
    sky.bindings = {frame_binding};
    sky.cull = rhi::CullMode::None;
    sky.depth_test = false;
    sky.color_format = kHdrFormat;
    sky_pipeline_ = device_.create_graphics_pipeline(sky);

    rhi::GraphicsPipelineDesc main;
    main.vertex_shader = shaders_[kMainVS];
    main.fragment_shader = shaders_[kMainFS];
    main.vertex_stride = sizeof(Vertex);
    main.attributes = attributes;
    main.push_constant_size = sizeof(Mat4);
    main.bindings = {frame_binding, {1, rhi::BindingType::SampledTexture, rhi::Filter::Nearest}};
    main.color_format = kHdrFormat;
    main_pipeline_ = device_.create_graphics_pipeline(main);
    main.cull = rhi::CullMode::None;
    main_two_sided_pipeline_ = device_.create_graphics_pipeline(main);

    rhi::GraphicsPipelineDesc post;
    post.vertex_shader = shaders_[kPostVS];
    post.fragment_shader = shaders_[kPostFS];
    post.bindings = {frame_binding, {2, rhi::BindingType::SampledTexture, rhi::Filter::Linear}};
    post.cull = rhi::CullMode::None;
    post.depth_test = false;
    post.depth_format = rhi::Format::Undefined;
    post_pipeline_ = device_.create_graphics_pipeline(post);

    shadow_map_ = device_.create_texture({shadow_size_, shadow_size_, rhi::Format::D32_Float, true, true});
    for (u32 i = 0; i < device_.frames_in_flight(); ++i) {
        frame_buffers_.push_back(device_.create_buffer({sizeof(FrameData), rhi::BufferUsage::Uniform, nullptr}));
    }

    valid_ = shadow_pipeline_.valid() && sky_pipeline_.valid() && main_pipeline_.valid() &&
             main_two_sided_pipeline_.valid() && post_pipeline_.valid() && shadow_map_.valid() &&
             std::ranges::all_of(frame_buffers_, [](rhi::BufferHandle b) { return b.valid(); });
    if (!valid_) {
        NOVA_ERROR("Renderer", "world renderer initialisation failed");
    }
}

WorldRenderer::~WorldRenderer() {
    for (const GpuMesh& m : meshes_) {
        device_.destroy(m.vertices);
        device_.destroy(m.indices);
    }
    for (rhi::BufferHandle b : frame_buffers_) {
        device_.destroy(b);
    }
    device_.destroy(hdr_target_);
    device_.destroy(shadow_map_);
    for (rhi::PipelineHandle p :
         {shadow_pipeline_, sky_pipeline_, main_pipeline_, main_two_sided_pipeline_, post_pipeline_}) {
        device_.destroy(p);
    }
    for (rhi::ShaderHandle s : shaders_) {
        device_.destroy(s);
    }
}

MeshId WorldRenderer::upload(const MeshData& mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty()) {
        return {};
    }
    GpuMesh gpu;
    gpu.vertices = device_.create_buffer(
        {mesh.vertices.size() * sizeof(Vertex), rhi::BufferUsage::Vertex, mesh.vertices.data()});
    gpu.indices =
        device_.create_buffer({mesh.indices.size() * sizeof(u32), rhi::BufferUsage::Index, mesh.indices.data()});
    gpu.index_count = static_cast<u32>(mesh.indices.size());
    if (!gpu.vertices.valid() || !gpu.indices.valid()) {
        device_.destroy(gpu.vertices);
        device_.destroy(gpu.indices);
        return {};
    }
    meshes_.push_back(gpu);
    return {static_cast<u32>(meshes_.size())};
}

bool WorldRenderer::ensure_hdr_target(rhi::Extent extent) {
    if (hdr_target_.valid() && extent.width == hdr_extent_.width && extent.height == hdr_extent_.height) {
        return true;
    }
    device_.destroy(hdr_target_);
    hdr_target_ = device_.create_texture({extent.width, extent.height, kHdrFormat, true, true});
    hdr_extent_ = extent;
    return hdr_target_.valid();
}

bool WorldRenderer::render_frame(const Camera& camera, const WorldSettings& settings,
                                 std::span<const WorldDrawItem> items) {
    if (!valid_) {
        return false;
    }
    const rhi::Extent extent = device_.back_buffer_extent();
    if (extent.width == 0 || extent.height == 0 || !ensure_hdr_target(extent)) {
        return false;
    }
    rhi::CommandList* cmd = device_.begin_frame();
    if (!cmd) {
        return false;
    }

    // ---- per-frame constants
    const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
    const Vec3 sun = math::normalize(settings.sun_direction);
    const Vec3 forward = math::normalize(camera.target - camera.position);
    const Vec3 right = math::normalize(math::cross(forward, camera.up));
    const Vec3 up = math::cross(right, forward);
    const f32 tan_half = std::tan(camera.fov_y * 0.5f);

    const f32 r = settings.shadow_radius;
    const Mat4 light_view = math::look_at(settings.shadow_center + sun * (r * 3.0f), settings.shadow_center,
                                          std::abs(sun.y) > 0.99f ? Vec3{0, 0, 1} : Vec3{0, 1, 0});
    const Mat4 light_proj = math::orthographic(-r, r, -r, r, 0.1f, r * 6.0f);

    FrameData fd{};
    fd.view_proj = camera.projection(aspect) * camera.view();
    fd.light_view_proj = light_proj * light_view;
    set4(fd.camera_pos, camera.position, settings.time);
    set4(fd.camera_right, right, tan_half * aspect);
    set4(fd.camera_up, up, tan_half);
    set4(fd.camera_forward, forward, settings.exposure);
    set4(fd.sun_dir, sun, 1.0f / static_cast<f32>(shadow_size_));
    set4(fd.sun_color, settings.sun_color * settings.sun_intensity, 0.0f);
    set4(fd.sky_zenith, settings.sky_zenith, settings.fog_density);
    set4(fd.sky_horizon, settings.sky_horizon, settings.fog_height_falloff);
    set4(fd.ground_color, settings.ground_color, settings.ambient);

    // One buffer per frame in flight: begin_frame() has waited for the GPU to finish this slot.
    const rhi::BufferHandle frame_buffer = frame_buffers_[frame_counter_ % frame_buffers_.size()];
    ++frame_counter_;
    device_.update_buffer(frame_buffer, &fd, sizeof(fd));
    const rhi::ResourceBinding frame_res{0, frame_buffer, {}};

    stats_ = {};
    auto draw_mesh = [&](const WorldDrawItem& item) {
        if (item.mesh.id == 0 || item.mesh.id > meshes_.size()) {
            return;
        }
        const GpuMesh& mesh = meshes_[item.mesh.id - 1];
        cmd->push_constants(&item.transform, sizeof(Mat4));
        cmd->bind_vertex_buffer(mesh.vertices);
        cmd->bind_index_buffer(mesh.indices);
        cmd->draw_indexed(mesh.index_count);
        ++stats_.draw_calls;
        stats_.triangles += mesh.index_count / 3;
    };

    // ---- 1. sun shadow map
    {
        rhi::RenderPassDesc pass;
        pass.back_buffer = false;
        pass.depth = shadow_map_;
        cmd->begin_render_pass(pass);
        cmd->bind_pipeline(shadow_pipeline_);
        cmd->bind_resources({&frame_res, 1});
        for (const WorldDrawItem& item : items) {
            if (item.casts_shadow) {
                draw_mesh(item);
            }
        }
        cmd->end_render_pass();
    }

    // ---- 2. sky + lit geometry into HDR
    {
        rhi::RenderPassDesc pass;
        pass.back_buffer = false;
        pass.color = hdr_target_;
        cmd->begin_render_pass(pass);

        cmd->bind_pipeline(sky_pipeline_);
        cmd->bind_resources({&frame_res, 1});
        cmd->draw(3);

        const rhi::ResourceBinding main_res[] = {frame_res, {1, {}, shadow_map_}};
        for (bool two_sided : {false, true}) {
            cmd->bind_pipeline(two_sided ? main_two_sided_pipeline_ : main_pipeline_);
            cmd->bind_resources(main_res);
            for (const WorldDrawItem& item : items) {
                if (item.two_sided == two_sided) {
                    draw_mesh(item);
                }
            }
        }
        cmd->end_render_pass();
    }

    // ---- 3. tonemap + grade into the back buffer
    {
        rhi::RenderPassDesc pass;
        pass.default_depth = false;
        cmd->begin_render_pass(pass);
        cmd->bind_pipeline(post_pipeline_);
        const rhi::ResourceBinding post_res[] = {frame_res, {2, {}, hdr_target_}};
        cmd->bind_resources(post_res);
        cmd->draw(3);
        cmd->end_render_pass();
    }

    device_.end_frame();
    return true;
}

} // namespace nova::renderer
