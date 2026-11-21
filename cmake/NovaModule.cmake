# Shared helpers for engine modules.
#
# nova_add_module(<name> SOURCES ... [PUBLIC_DEPS ...] [PRIVATE_DEPS ...])
#   Creates static library `nova_<name>` with alias `Nova::<name>`.
#   Public headers live in <module>/include/nova/<name>/ and are the module's only
#   contract; everything under src/ is private (see docs/adr/0003).

function(nova_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
        if(NOVA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
            -Wno-sign-conversion)
        if(NOVA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
endfunction()

function(nova_add_module name)
    cmake_parse_arguments(ARG "" "" "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS" ${ARGN})
    set(target nova_${name})
    add_library(${target} STATIC ${ARG_SOURCES})
    add_library(Nova::${name} ALIAS ${target})
    target_include_directories(${target}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_compile_definitions(${target} PUBLIC $<$<CONFIG:Debug>:NOVA_ENABLE_ASSERTS=1>)
    if(ARG_PUBLIC_DEPS)
        target_link_libraries(${target} PUBLIC ${ARG_PUBLIC_DEPS})
    endif()
    if(ARG_PRIVATE_DEPS)
        target_link_libraries(${target} PRIVATE ${ARG_PRIVATE_DEPS})
    endif()
    nova_set_warnings(${target})
endfunction()
