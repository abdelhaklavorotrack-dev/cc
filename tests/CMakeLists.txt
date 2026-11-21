add_executable(nova_tests
    main.cpp
    core_tests.cpp
    memory_tests.cpp
    jobs_tests.cpp
    platform_tests.cpp)
target_link_libraries(nova_tests PRIVATE
    Nova::core Nova::memory Nova::jobs Nova::platform doctest::doctest)
nova_set_warnings(nova_tests)

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(nova_tests)

# Smoke test: the sandbox frame loop runs end-to-end without a display.
if(TARGET nova_sandbox)
    add_test(NAME sandbox.headless_smoke COMMAND nova_sandbox --headless --frames 120)
endif()
