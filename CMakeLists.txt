cmake_minimum_required(VERSION 3.25)

project(NovaEngine
    VERSION 0.1.0
    DESCRIPTION "NOVA Engine - modular 3D game engine"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(NOVA_BUILD_TESTS "Build unit tests" ON)
option(NOVA_BUILD_SANDBOX "Build the sandbox application" ON)
option(NOVA_WITH_SDL3 "Build the SDL3 platform backend (otherwise headless only)" ON)
option(NOVA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

include(NovaModule)
include(NovaDependencies)

add_subdirectory(engine/core)
add_subdirectory(engine/memory)
add_subdirectory(engine/jobs)
add_subdirectory(engine/platform)

if(NOVA_BUILD_SANDBOX)
    add_subdirectory(apps/sandbox)
endif()

if(NOVA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
