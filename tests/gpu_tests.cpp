// GPU tests: render offscreen through the RHI and inspect pixels. Runs on any Vulkan 1.3
// device, including lavapipe in CI. Exits 77 (CTest "skipped") when no device exists.

#define DOCTEST_CONFIG_IMPLEMENT
#include "nova/core/assert.hpp"
#include "nova/core/log.hpp"
#include "nova/renderer/renderer.hpp"
#include "nova/rhi/rhi.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using namespace nova;

namespace {

constexpr u32 kSize = 64;

std::unique_ptr<rhi::Device> make_device(u32 w = kSize, u32 h = kSize) {
    rhi::DeviceDesc desc;
    desc.width = w;
    desc.height = h;
    desc.validation = true; // used when installed; tests then require zero errors
    return rhi::Device::create(desc);
}

u8 to_srgb8(f32 linear) {
    const f32 s = linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    return static_cast<u8>(std::lround(std::clamp(s, 0.0f, 1.0f) * 255.0f));
}

struct Pixel {
    int r, g, b, a;
};

Pixel pixel_at(const std::vector<u8>& rgba, u32 width, u32 x, u32 y) {
    const usize i = (static_cast<usize>(y) * width + x) * 4;
    return {rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]};
}

bool close_to(const Pixel& p, int r, int g, int b, int tol = 3) {
    return std::abs(p.r - r) <= tol && std::abs(p.g - g) <= tol && std::abs(p.b - b) <= tol;
}

[[noreturn]] void throwing_assert_handler(std::string_view expr, std::string_view msg, const std::source_location&) {
    throw std::logic_error(std::string(msg) + " (" + std::string(expr) + ")");
}

} // namespace

TEST_CASE("gpu: clear colour reaches the back buffer in sRGB") {
    auto device = make_device();
    REQUIRE(device);
    rhi::CommandList* cmd = device->begin_frame();
    REQUIRE(cmd);
    rhi::ClearValues clear;
    clear.color[0] = 1.0f;
    clear.color[1] = 0.5f;
    clear.color[2] = 0.0f;
    cmd->begin_render_pass(clear);
    cmd->end_render_pass();
    device->end_frame();

    std::vector<u8> rgba;
    REQUIRE(device->read_back(rgba));
    REQUIRE(rgba.size() == kSize * kSize * 4);
    CHECK(close_to(pixel_at(rgba, kSize, 0, 0), 255, to_srgb8(0.5f), 0, 1));
    CHECK(close_to(pixel_at(rgba, kSize, kSize - 1, kSize - 1), 255, to_srgb8(0.5f), 0, 1));
    CHECK(device->validation_error_count() == 0);
}

TEST_CASE("gpu: lit cube renders front face with correct winding and orientation") {
    auto device = make_device();
    REQUIRE(device);
    {
        renderer::Renderer renderer(*device);
        REQUIRE(renderer.valid());
        const auto cube = renderer.upload(renderer::make_cube(1.0f));
        REQUIRE(cube.id != 0);

        renderer::Camera camera;
        camera.position = {0, 0, 3};
        camera.target = {0, 0, 0};
        renderer::DirectionalLight light;
        light.direction = {0, 0, 1}; // straight at the +Z face
        light.ambient = 0.0f;

        // Big cube at the centre, small cube above it: checks culling/winding and that +Y is up.
        const renderer::DrawItem items[] = {
            {cube, math::Mat4::identity(), {0.5f, 0.5f, 0.5f}},
            {cube, math::translate({0, 1.1f, 0}) * math::scale({0.3f, 0.3f, 0.3f}), {0.0f, 1.0f, 0.0f}},
        };
        const math::Vec3 clear{0.0f, 0.0f, 1.0f};
        for (int i = 0; i < 3; ++i) { // several frames exercise frames-in-flight and timestamps
            REQUIRE(renderer.render_frame(camera, light, items, clear));
        }
        CHECK(renderer.stats().draw_calls == 2);
        CHECK(renderer.stats().triangles == 24);

        std::vector<u8> rgba;
        REQUIRE(device->read_back(rgba));
        const u8 grey = to_srgb8(0.5f);
        // If back-face culling used the wrong winding we would see the unlit -Z face (black).
        CHECK(close_to(pixel_at(rgba, kSize, kSize / 2, kSize / 2), grey, grey, grey));
        CHECK(close_to(pixel_at(rgba, kSize, 1, 1), 0, 0, 255, 1)); // background
        // Small green cube must be in the upper part of the image.
        CHECK(close_to(pixel_at(rgba, kSize, kSize / 2, kSize * 10 / 64), 0, 255, 0));
        CHECK(close_to(pixel_at(rgba, kSize, kSize / 2, kSize * 57 / 64), 0, 0, 255, 1));
    }
    if (device->info().gpu_timing) {
        CHECK(device->last_gpu_frame_ms() >= 0.0);
    }
    CHECK(device->validation_error_count() == 0);
}

TEST_CASE("gpu: offscreen target resizes") {
    auto device = make_device();
    REQUIRE(device);
    device->resize(32, 16);
    CHECK(device->back_buffer_extent().width == 32);
    CHECK(device->back_buffer_extent().height == 16);
    rhi::CommandList* cmd = device->begin_frame();
    REQUIRE(cmd);
    cmd->begin_render_pass({});
    cmd->end_render_pass();
    device->end_frame();
    std::vector<u8> rgba;
    REQUIRE(device->read_back(rgba));
    CHECK(rgba.size() == 32 * 16 * 4);
    CHECK(device->validation_error_count() == 0);
}

TEST_CASE("gpu: buffers can be created, updated and destroyed") {
    auto device = make_device();
    REQUIRE(device);
    const u32 data[4] = {1, 2, 3, 4};
    const auto buf = device->create_buffer({sizeof(data), rhi::BufferUsage::Vertex, data});
    REQUIRE(buf.valid());
    device->update_buffer(buf, data, sizeof(u32), 4);
    CHECK_THROWS_AS(device->update_buffer(buf, data, sizeof(data), 4), std::logic_error); // out of range
    device->destroy(buf);
    device->destroy(buf); // second destroy is a no-op
    CHECK(device->validation_error_count() == 0);
}

int main(int argc, char** argv) {
    nova::set_assert_handler(&throwing_assert_handler);
    if (!make_device()) {
        // CI sets NOVA_REQUIRE_GPU where lavapipe is installed, so a broken driver can't skip silently.
        if (const char* req = std::getenv("NOVA_REQUIRE_GPU"); req && *req) {
            std::fprintf(stderr, "NOVA_REQUIRE_GPU is set but no Vulkan 1.3 device is available\n");
            return 1;
        }
        std::fprintf(stderr, "no Vulkan 1.3 device available; skipping GPU tests\n");
        return 77;
    }
    log::set_level(log::Level::Warn);
    doctest::Context context(argc, argv);
    return context.run();
}
