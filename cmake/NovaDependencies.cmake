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

# ---- Rendering: Vulkan headers + volk loader (no link-time dependency on the Vulkan SDK) ----
if(NOVA_WITH_VULKAN)
    FetchContent_Declare(VulkanHeaders
        URL https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.335.0.tar.gz
        URL_HASH SHA256=269e95cc5138ea0a0d52fcb0ee19102add2560fedf5a43b1b5c17780c2775764
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR do-not-add)
    FetchContent_Declare(volk
        URL https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.335.0.tar.gz
        URL_HASH SHA256=ca2beb1ab9bf272895bc0152ef29b437d178d837bae4a76d0022b7bbafb3c483
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR do-not-add)
    FetchContent_MakeAvailable(VulkanHeaders volk)

    add_library(nova_vulkan_headers INTERFACE)
    target_include_directories(nova_vulkan_headers SYSTEM INTERFACE ${vulkanheaders_SOURCE_DIR}/include)
    target_compile_definitions(nova_vulkan_headers INTERFACE VK_NO_PROTOTYPES)

    add_library(nova_volk STATIC ${volk_SOURCE_DIR}/volk.c)
    target_include_directories(nova_volk SYSTEM PUBLIC ${volk_SOURCE_DIR})
    target_link_libraries(nova_volk PUBLIC nova_vulkan_headers ${CMAKE_DL_LIBS})

    # Shader compiler: prefer slangc from PATH / Vulkan SDK, else the pinned release binary.
    find_program(NOVA_SLANGC slangc HINTS $ENV{VULKAN_SDK}/bin)
    if(NOT NOVA_SLANGC)
        if(CMAKE_HOST_WIN32)
            set(_slang_url https://github.com/shader-slang/slang/releases/download/v2026.22.1/slang-2026.22.1-windows-x86_64.zip)
            set(_slang_hash e72344f7b0323cec8b039c534caaf4680e7ba629489fc796ea4c7f7adb7deb68)
            set(_slang_exe slangc.exe)
        else()
            set(_slang_url https://github.com/shader-slang/slang/releases/download/v2026.22.1/slang-2026.22.1-linux-x86_64.tar.gz)
            set(_slang_hash 37156099ebaabf62cd92522eba55eb93b8663e65ea3be618ce015d9a8edda6bc)
            set(_slang_exe slangc)
        endif()
        FetchContent_Declare(slang_bin URL ${_slang_url} URL_HASH SHA256=${_slang_hash}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE SOURCE_SUBDIR do-not-add)
        FetchContent_MakeAvailable(slang_bin)
        set(NOVA_SLANGC ${slang_bin_SOURCE_DIR}/bin/${_slang_exe} CACHE FILEPATH "" FORCE)
    endif()
    message(STATUS "NOVA: shader compiler ${NOVA_SLANGC}")
endif()
