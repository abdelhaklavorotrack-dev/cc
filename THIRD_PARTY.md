# Third-party software

| Name | Version | License | Used by | Notes |
|------|---------|---------|---------|-------|
| [SDL3](https://github.com/libsdl-org/SDL) | 3.2.8 | zlib | `Nova::platform` (private) | Window/input backend, static, optional (`NOVA_WITH_SDL3`) |
| [doctest](https://github.com/doctest/doctest) | 2.4.11 | MIT | `tests/` only | Header-only unit test framework |

Pinned archives and SHA-256 hashes are in `cmake/NovaDependencies.cmake`. Policy: `docs/adr/0006-third-party-dependencies.md`.
