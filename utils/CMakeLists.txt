add_library(LayersUtils STATIC
    utils.cpp
    utils.hpp)

target_include_directories(LayersUtils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(LayersUtils PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (OPENCL_LAYERS_BUILD_TESTING)
add_executable(print_settings_location print_settings_location.cpp)
target_link_libraries(print_settings_location PRIVATE LayersUtils LayersCommon)

add_test(NAME test_settings_locations
         COMMAND ${CMAKE_COMMAND}
            -DBINARY_DIR=$<TARGET_FILE_DIR:print_settings_location>
            -DTARGET_FILE=$<TARGET_FILE:print_settings_location>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test_settings_location.cmake)
endif()