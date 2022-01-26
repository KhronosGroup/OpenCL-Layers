add_library (CLObjectLifetimeLayer SHARED
    object_lifetime.cpp
    $<$<CXX_COMPILER_ID:MSVC>:object_lifetime.def>
    $<$<CXX_COMPILER_ID:GNU>:object_lifetime.map>
)

target_link_libraries (CLObjectLifetimeLayer PRIVATE LayersUtils LayersCommon)

if (NOT WIN32 AND NOT APPLE)
    set_target_properties (CLObjectLifetimeLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/object_lifetime.map")
endif ()

if (OPENCL_LAYERS_BUILD_TESTING)
    add_executable (CLObjectLifetimeLayerTest object_lifetime_test.c)

    target_link_libraries (CLObjectLifetimeLayerTest
        PRIVATE
            LayersCommon
            OpenCL::OpenCL
    )

    add_test (
        NAME CLObjectLifetimeLayerTest
        COMMAND "${CMAKE_COMMAND}"
            -DCOMMAND=$<TARGET_FILE:CLObjectLifetimeLayerTest>
            -DEXPECTED_OUTPUT=${CMAKE_CURRENT_LIST_DIR}/object_lifetime_test_expected.regex
            -DEXTRA_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/CLObjectLifetimeLayerTest.log
            -DEXPECTED_EXTRA_OUTPUT=${CMAKE_CURRENT_SOURCE_DIR}/expected_layer_log.regex
            -P ${CMAKE_CURRENT_LIST_DIR}/object_lifetime_test.cmake
    )
    set(TEST_ENVIRONMENT
            OPENCL_LAYERS=$<TARGET_FILE:CLObjectLifetimeLayer>
            CL_OBJECT_LIFETIME_TRANSPARENT=ON
            CL_OBJECT_LIFETIME_LOG_SINK=file
            CL_OBJECT_LIFETIME_LOG_FILENAME=CLObjectLifetimeLayerTest.log)
    set_tests_properties (CLObjectLifetimeLayerTest
        PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}"
    )
endif ()

set (INSTALL_TARGETS CLObjectLifetimeLayer)
set (BUILD_TARGETS ${INSTALL_TARGETS})

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
