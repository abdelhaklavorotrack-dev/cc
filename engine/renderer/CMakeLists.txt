nova_add_module(renderer
    SOURCES
        src/mesh.cpp
        src/renderer.cpp
        src/world_renderer.cpp
    PUBLIC_DEPS
        Nova::math
        Nova::rhi)

nova_add_shader(nova_renderer shaders/forward.slang forward_spv)
nova_add_shader(nova_renderer shaders/world.slang world_spv)
