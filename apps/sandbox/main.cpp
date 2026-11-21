// Foundation sandbox: opens a window, runs the frame loop, exercises the job system and
// per-frame arena, and reports frame-time statistics once per second.
//
//   nova_sandbox [--headless] [--frames N]

#include "nova/core/log.hpp"
#include "nova/core/time.hpp"
#include "nova/jobs/job_system.hpp"
#include "nova/memory/allocators.hpp"
#include "nova/platform/window.hpp"

#include <atomic>
#include <charconv>
#include <cstring>
#include <string_view>

using namespace nova;

int main(int argc, char** argv) {
    platform::WindowDesc desc;
    desc.title = "NOVA Sandbox";
    u64 max_frames = 0; // 0 = until the window closes

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--headless") {
            desc.backend = platform::Backend::Headless;
        } else if (arg == "--frames" && i + 1 < argc) {
            const char* v = argv[++i];
            std::from_chars(v, v + std::strlen(v), max_frames);
        }
    }
    if (desc.backend == platform::Backend::Headless && max_frames == 0) {
        max_frames = 600; // headless has no close button
    }

    auto window = platform::create_window(desc);
    if (!window) {
        NOVA_FATAL("Sandbox", "could not create a window");
        return 1;
    }

    jobs::JobSystem job_system;
    memory::LinearAllocator frame_arena(1u << 20);
    platform::Input input;
    FrameStats stats;
    Stopwatch frame_timer;
    Stopwatch report_timer;
    constexpr u32 kItems = 64 * 1024;

    NOVA_INFO("Sandbox", "running with {} workers, press Escape to quit", job_system.worker_count());

    for (u64 frame = 0; max_frames == 0 || frame < max_frames; ++frame) {
        input.begin_frame();
        if (!window->poll_events(input) || input.was_pressed(platform::Key::Escape)) {
            break;
        }

        // Stand-in for simulation work: fill a per-frame buffer in parallel.
        frame_arena.reset();
        auto* values = static_cast<f32*>(frame_arena.allocate(kItems * sizeof(f32), alignof(f32)));
        std::atomic<u64> checksum{0};
        job_system.parallel_for(kItems, 4096, [&](u32 begin, u32 end) {
            u64 local = 0;
            for (u32 i = begin; i < end; ++i) {
                values[i] = static_cast<f32>(i) * 0.5f;
                local += i;
            }
            checksum.fetch_add(local, std::memory_order_relaxed);
        });

        stats.record(frame_timer.lap_ms());
        if (report_timer.elapsed_ms() >= 1000.0) {
            report_timer.reset();
            const auto s = stats.summarize();
            NOVA_INFO("Sandbox", "frame avg {:.3f} ms | p99 {:.3f} ms | {:.0f} fps | 1% low {:.0f} fps",
                      s.avg_ms, s.p99_ms, s.avg_fps, s.one_percent_low_fps);
        }
    }

    const auto s = stats.summarize();
    NOVA_INFO("Sandbox", "exit: {} frames sampled, avg {:.3f} ms, p99 {:.3f} ms, arena peak {} bytes",
              s.frame_count, s.avg_ms, s.p99_ms, frame_arena.peak());
    return 0;
}
