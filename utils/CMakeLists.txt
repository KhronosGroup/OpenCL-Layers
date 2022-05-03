add_library(LayersUtils STATIC
    utils.cpp
    utils.hpp)
target_include_directories(LayersUtils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LayersUtils PUBLIC LayersCommon)
set_target_properties(LayersUtils PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (LAYERS_BUILD_TESTS)
add_executable(print_settings_location print_settings_location.cpp)
target_link_libraries(print_settings_location PRIVATE LayersUtils LayersCommon)

include(test_settings_location.cmake)
include(test_settings.cmake)

endif()
