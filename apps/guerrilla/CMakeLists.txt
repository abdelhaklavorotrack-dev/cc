add_executable(nova_guerrilla main.cpp viewmodel.cpp)
target_link_libraries(nova_guerrilla PRIVATE Nova::core Nova::platform Nova::renderer nova_village_scene)
nova_set_warnings(nova_guerrilla)
