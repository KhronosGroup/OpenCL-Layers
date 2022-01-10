add_library (PrintLayer SHARED
    icd_print_layer.c
    icd_print_layer.h
    icd_print_layer_generated.c
    $<$<CXX_COMPILER_ID:MSVC>:icd_print_layer.def>
    $<$<CXX_COMPILER_ID:GNU>:icd_print_layer.map>
)

target_link_libraries (PrintLayer PRIVATE LayersCommon)

if (NOT WIN32 AND NOT APPLE)
    set_target_properties (PrintLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/icd_print_layer.map")
endif ()

install (
    TARGETS PrintLayer
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if (OPENCL_LAYERS_BUILD_TESTING)
    add_executable (PrintLayerTest icd_print_layer_test.c)

    target_link_libraries (PrintLayerTest
        PRIVATE
            LayersCommon
            OpenCL::OpenCL
    )

    add_test (
        NAME PrintLayerTest
        COMMAND PrintLayerTest
    )
    file(READ icd_print_layer_test.regex PRINT_LAYER_REGEX)
    set_tests_properties (PrintLayerTest
        PROPERTIES
            PASS_REGULAR_EXPRESSION ${PRINT_LAYER_REGEX}
            ENVIRONMENT OPENCL_LAYERS=$<TARGET_FILE:PrintLayer>
    )
endif ()