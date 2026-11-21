# NOVA Engine

A modular, Windows-first 3D game engine targeting FPS/TPS and open-world games. The engine is built in
small, independently testable modules with documented contracts, and every performance claim is backed by a
benchmark.

**Current phase: 1 — Foundation** (build, CI, window & input, logging, memory, jobs).
See the roadmap below and the decisions in [`docs/adr/`](docs/adr).

## Building

Requirements: CMake ≥ 3.25, Ninja, and a C++23 compiler (Visual Studio 2022 17.8+ or GCC 13+).
Dependencies (SDL3, doctest) are fetched automatically at pinned versions.

```sh
# Windows (Developer PowerShell for VS 2022)
cmake --preset windows-msvc
cmake --build --preset windows-msvc-debug
ctest --preset windows-msvc-debug

# Linux
cmake --preset linux-gcc
cmake --build --preset linux-gcc-debug
ctest --preset linux-gcc-debug

# Without SDL3 (CI, servers)
cmake --preset headless && cmake --build --preset headless-debug && ctest --preset headless-debug
```

Run the sandbox: `build/<preset>/apps/sandbox/Debug/nova_sandbox` (`--headless`, `--frames N`). It opens a
window, runs the frame loop on the job system and logs avg / p99 frame time and 1% low FPS every second.

## Layout

```
engine/
  core/       types, logging, assertions, timing & frame statistics
  memory/     linear (frame) and pool allocators
  jobs/       worker pool, wait groups, parallel_for
  platform/   window + input (SDL3 backend, headless backend)
apps/sandbox/ foundation smoke-test application
tests/        doctest unit tests, registered with CTest
docs/adr/     architecture decision records
cmake/        build helpers and pinned dependencies
```

Module rules (public headers only under `include/nova/<module>/`, one-way layering, no third-party types in
public headers) are in [ADR 0003](docs/adr/0003-module-architecture.md).

## Roadmap

1. **Foundation** — build/CI, window & input, logging, memory, jobs. *(in progress)*
2. **Rendering Core** — Vulkan RHI, simple 3D scene, CPU/GPU frame timing ([ADR 0005](docs/adr/0005-graphics-api-vulkan-slang.md)).
3. **Scene & Editor** — ECS, scene save/load, Hierarchy, Inspector, gizmos.
4. **Smart Rendering** — culling, LOD/HLOD, instancing, render graph, streaming (with documented gains).
5. **Advanced Graphics** — shadows, GI, fog, SSR, optional ray tracing.
6. **Physics & Animation** — Jolt behind an interface; FPS/TPS character interacting with the world.
7. **Production** — audio, packaging, tools, sample game.

First vertical slice: create a project, import a glTF model, place it in a scene, apply PBR, add a light,
camera and simple physics, save the scene and run it in a standalone Runtime.

## Contributing

Each task states its scope, the files it owns, the API contract it touches and its success criteria. Work
happens on a branch; a change merges only after it builds with warnings-as-errors, passes `ctest` (including
sanitizer jobs in CI) and is reviewed. Changes to public headers or cross-module rules need an ADR
([ADR 0001](docs/adr/0001-record-architecture-decisions.md)). Code style: `.clang-format`.
