add_library (CLObjectLifetimeLayer SHARED
    object_lifetime.cpp
    $<$<CXX_COMPILER_ID:MSVC>:object_lifetime.def>
    $<$<CXX_COMPILER_ID:GNU>:object_lifetime.map>
)

target_link_libraries (CLObjectLifetimeLayer PRIVATE LayersUtils LayersCommon)

if (NOT WIN32 AND NOT APPLE)
    set_target_properties (CLObjectLifetimeLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/object_lifetime.map")
endif ()

set (INSTALL_TARGETS CLObjectLifetimeLayer)
set (BUILD_TARGETS ${INSTALL_TARGETS})

if (LAYERS_BUILD_TESTS)
    add_library (CLObjectLifetimeICD SHARED
        object_lifetime_test_icd.hpp
        object_lifetime_test_icd.cpp
        object_lifetime_test_icd_surface.cpp
        $<$<CXX_COMPILER_ID:MSVC>:object_lifetime_test_icd.def>
        $<$<CXX_COMPILER_ID:GNU>:object_lifetime_test_icd.map>
    )
    target_link_libraries (CLObjectLifetimeICD
        PRIVATE
            LayersCommon
            LayersUtils
            OpenCL::Headers
    )

    add_executable (CLObjectLifetimeLayerTest object_lifetime_test.cpp)

    target_link_libraries (CLObjectLifetimeLayerTest
        PRIVATE
            LayersCommon
            OpenCL::OpenCL
    )

    function (add_layer_test OPENCL_VERSION)
        cmake_parse_arguments(PARSE_ARGV 1 ARG "REPORT_IMPLICIT_REF_COUNT;ALLOW_USING_RELEASED_OBJECTS;ALLOW_USING_INACCESSIBLE_OBJECTS" "" "")

        set (TEST_ENVIRONMENT
            OPENCL_LAYERS=$<TARGET_FILE:CLObjectLifetimeLayer>
            CL_OBJECT_LIFETIME_TRANSPARENT=1
            CL_OBJECT_LIFETIME_LOG_SINK=file
            CL_OBJECT_LIFETIME_LOG_FILENAME=CLObjectLifetimeLayerTest.log
            OCL_ICD_FILENAMES=$<TARGET_FILE:CLObjectLifetimeICD>
            OBJECT_LIFETIME_ICD_VERSION=${OPENCL_VERSION}
        )
        set (TEST_NAME "ObjectLifetime-CL${OPENCL_VERSION}")

        if (ARG_REPORT_IMPLICIT_REF_COUNT)
          list(APPEND TEST_ENVIRONMENT REPORT_IMPLICIT_REF_COUNT_TO_USER=1)
          string (APPEND TEST_NAME "-ReportImplicitRefCount")
        endif ()
        if (ARG_ALLOW_USING_RELEASED_OBJECTS)
          list (APPEND TEST_ENVIRONMENT ALLOW_USING_RELEASED_OBJECTS=1)
          string (APPEND TEST_NAME "-AllowUsingReleasedObjects")
        endif()
        if (ARG_ALLOW_USING_INACCESSIBLE_OBJECTS)
          list (APPEND TEST_ENVIRONMENT ALLOW_USING_INACCESSIBLE_OBJECTS)
          string (APPEND TEST_NAME "-AllowUsingInaccessibleObjects")
        endif()

        add_test (
            NAME ${TEST_NAME}
            COMMAND "${CMAKE_COMMAND}"
                -DCOMMAND=$<TARGET_FILE:CLObjectLifetimeLayerTest>
                -DEXTRA_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/CLObjectLifetimeLayerTest.log
                -DEXPECTED_EXTRA_OUTPUT=${CMAKE_CURRENT_SOURCE_DIR}/expected_layer_log.regex
                -P ${CMAKE_CURRENT_LIST_DIR}/object_lifetime_test.cmake
        )
        set_tests_properties (${TEST_NAME} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
    endfunction()

    add_layer_test(200)
endif ()

list (APPEND BUILD_TARGETS
    CLObjectLifetimeICD
    CLObjectLifetimeLayerTest
)

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
