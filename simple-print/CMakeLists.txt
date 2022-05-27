add_library (PrintLayer SHARED
    icd_print_layer.c
    icd_print_layer.h
    icd_print_layer_generated.c
    $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:MSVC,Clang>>:icd_print_layer.def>
    $<$<CXX_COMPILER_ID:GNU>:icd_print_layer.map>
)

target_link_libraries (PrintLayer PRIVATE LayersCommon)

if (NOT WIN32 AND NOT APPLE)
    set_target_properties (PrintLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/icd_print_layer.map")
endif ()

if (LAYERS_BUILD_TESTS)
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

set (INSTALL_TARGETS PrintLayer)
set (BUILD_TARGETS ${INSTALL_TARGETS})
if (LAYERS_BUILD_TESTS)
    list (APPEND BUILD_TARGETS PrintLayerTest)
endif ()

set_target_properties (${BUILD_TARGETS}
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        FOLDER "Layers"
)
install (
    TARGETS ${INSTALL_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
