# Third-party dependencies, pinned by version and hash (see docs/adr/0006 and THIRD_PARTY.md).
include(FetchContent)

if(NOVA_WITH_SDL3)
    find_package(SDL3 3.2 CONFIG QUIET)
    if(NOT SDL3_FOUND)
        set(SDL_SHARED OFF CACHE BOOL "" FORCE)
        set(SDL_STATIC ON CACHE BOOL "" FORCE)
        set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
        set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(SDL3
            URL https://github.com/libsdl-org/SDL/releases/download/release-3.2.8/SDL3-3.2.8.tar.gz
            URL_HASH SHA256=13388fabb361de768ecdf2b65e52bb27d1054cae6ccb6942ba926e378e00db03
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_MakeAvailable(SDL3)
    endif()
endif()

if(NOVA_BUILD_TESTS)
    FetchContent_Declare(doctest
        URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz
        URL_HASH SHA256=632ed2c05a7f53fa961381497bf8069093f0d6628c5f26286161fbd32a560186
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        # Header-only use: skip doctest's own CMakeLists (its minimum version predates CMake 4).
        SOURCE_SUBDIR do-not-add)
    FetchContent_MakeAvailable(doctest)
    add_library(nova_doctest INTERFACE)
    add_library(doctest::doctest ALIAS nova_doctest)
    target_include_directories(nova_doctest SYSTEM INTERFACE ${doctest_SOURCE_DIR})
endif()
