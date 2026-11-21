#pragma once

// Render Hardware Interface: the engine's only view of the GPU. Backend types (Vulkan) never
// appear here (ADR 0003, 0005). Deliberately small: it grows with the renderer's needs.

#include "nova/core/types.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nova::platform {
class Window;
}

namespace nova::rhi {

// Typed 32-bit handles; 0 is invalid.
template <typename Tag>
struct Handle {
    u32 id = 0;
    [[nodiscard]] constexpr bool valid() const noexcept { return id != 0; }
    constexpr bool operator==(const Handle&) const noexcept = default;
};
using BufferHandle = Handle<struct BufferTag>;
using ShaderHandle = Handle<struct ShaderTag>;
using PipelineHandle = Handle<struct PipelineTag>;

enum class BufferUsage : u8 { Vertex, Index, Uniform };
enum class ShaderStage : u8 { Vertex, Fragment };
enum class VertexFormat : u8 { Float2, Float3, Float4 };
enum class CullMode : u8 { None, Back };

struct BufferDesc {
    usize size = 0;
    BufferUsage usage = BufferUsage::Vertex;
    const void* initial_data = nullptr; // optional, `size` bytes
};

struct ShaderDesc {
    std::span<const u32> spirv;
    std::string entry_point = "main";
    ShaderStage stage = ShaderStage::Vertex;
};

struct VertexAttribute {
    u32 location = 0;
    VertexFormat format = VertexFormat::Float3;
    u32 offset = 0;
};

struct GraphicsPipelineDesc {
    ShaderHandle vertex_shader;
    ShaderHandle fragment_shader;
    u32 vertex_stride = 0;
    std::vector<VertexAttribute> attributes;
    u32 push_constant_size = 0; // visible to vertex and fragment stages, <= 128 bytes
    CullMode cull = CullMode::Back;
    bool depth_test = true;
};

struct ClearValues {
    f32 color[4] = {0.0f, 0.0f, 0.0f, 1.0f}; // linear RGBA
    f32 depth = 1.0f;
};

// Records commands for the current frame. Valid between Device::begin_frame and end_frame.
class CommandList {
public:
    virtual ~CommandList() = default;

    // Renders to the frame's back buffer (swapchain image or offscreen target) plus depth.
    virtual void begin_render_pass(const ClearValues& clear) = 0;
    virtual void end_render_pass() = 0;

    virtual void bind_pipeline(PipelineHandle pipeline) = 0;
    virtual void bind_vertex_buffer(BufferHandle buffer, usize offset = 0) = 0;
    virtual void bind_index_buffer(BufferHandle buffer, usize offset = 0) = 0; // u32 indices
    virtual void push_constants(const void* data, u32 size) = 0;
    virtual void draw(u32 vertex_count, u32 first_vertex = 0) = 0;
    virtual void draw_indexed(u32 index_count, u32 first_index = 0, i32 vertex_offset = 0) = 0;
};

struct DeviceDesc {
    // nullptr => headless: render into an offscreen target of width x height.
    platform::Window* window = nullptr;
    u32 width = 1280;
    u32 height = 720;
    bool vsync = true;
    bool validation = false; // Vulkan validation layers + debug messenger, if installed
};

struct DeviceInfo {
    std::string backend;    // "Vulkan"
    std::string device_name;
    std::string api_version;
    bool software = false;  // CPU implementation (e.g. lavapipe)
    bool validation = false;
    bool gpu_timing = false; // timestamp queries supported
};

struct Extent {
    u32 width = 0;
    u32 height = 0;
};

class Device {
public:
    // Returns nullptr (logged) when no suitable GPU/driver is available.
    [[nodiscard]] static std::unique_ptr<Device> create(const DeviceDesc& desc);
    virtual ~Device() = default;

    [[nodiscard]] virtual const DeviceInfo& info() const noexcept = 0;
    [[nodiscard]] virtual Extent back_buffer_extent() const noexcept = 0;

    [[nodiscard]] virtual BufferHandle create_buffer(const BufferDesc& desc) = 0;
    // Host-visible write; caller must ensure the GPU is not reading the range this frame.
    virtual void update_buffer(BufferHandle buffer, const void* data, usize size, usize offset = 0) = 0;
    [[nodiscard]] virtual ShaderHandle create_shader(const ShaderDesc& desc) = 0;
    [[nodiscard]] virtual PipelineHandle create_graphics_pipeline(const GraphicsPipelineDesc& desc) = 0;
    // Destruction waits for the GPU to go idle (simple and safe; fine outside the hot path).
    virtual void destroy(BufferHandle handle) = 0;
    virtual void destroy(ShaderHandle handle) = 0;
    virtual void destroy(PipelineHandle handle) = 0;

    // Returns nullptr when the frame must be skipped (e.g. minimized window).
    [[nodiscard]] virtual CommandList* begin_frame() = 0;
    virtual void end_frame() = 0;
    // Call after the window size changes; the swapchain is also recreated lazily when stale.
    virtual void resize(u32 width, u32 height) = 0;
    virtual void wait_idle() = 0;

    // GPU time of the most recently completed frame in ms (negative until available).
    [[nodiscard]] virtual f64 last_gpu_frame_ms() const noexcept = 0;
    // Validation-layer errors reported so far (always 0 without validation).
    [[nodiscard]] virtual u32 validation_error_count() const noexcept = 0;

    // Headless only: copies the last rendered frame as tightly packed RGBA8 (sRGB) pixels.
    [[nodiscard]] virtual bool read_back(std::vector<u8>& rgba) = 0;
};

} // namespace nova::rhi
