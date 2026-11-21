add_executable(nova_village main.cpp)
target_link_libraries(nova_village PRIVATE Nova::core Nova::platform Nova::renderer)
nova_set_warnings(nova_village)
