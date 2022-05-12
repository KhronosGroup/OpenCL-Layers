function (add_layer_test_exe NAME SOURCE)
    add_executable (${NAME} ${SOURCE} object_lifetime_test.cpp)
    target_link_libraries (
        ${NAME}
        PRIVATE
            LayersCommon
            OpenCL::OpenCL
    )

    set_target_properties (${NAME}
        PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
            PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
            INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
            FOLDER "Layers"
    )
endfunction()

function (add_layer_test TEST_EXE OPENCL_VERSION)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "REPORT_IMPLICIT_REF_COUNT;ALLOW_USING_RELEASED_OBJECTS;ALLOW_USING_INACCESSIBLE_OBJECTS" "REGEX" "")

    set (TEST_ENVIRONMENT
        OPENCL_LAYERS=$<TARGET_FILE:CLObjectLifetimeLayer>
        CL_OBJECT_LIFETIME_TRANSPARENT=0
        CL_OBJECT_LIFETIME_LOG_SINK=file
        # OCL_ICD_FILENAMES=$<TARGET_FILE:CLObjectLifetimeICD>
        OBJECT_LIFETIME_ICD_VERSION=${OPENCL_VERSION}
    )
    set (TEST_NAME "ObjectLifetime-${TEST_EXE}-CL${OPENCL_VERSION}")

    set (TEST_INVOCATION $<TARGET_FILE:${TEST_EXE}>)

    if (ARG_REPORT_IMPLICIT_REF_COUNT)
      list (APPEND TEST_ENVIRONMENT REPORT_IMPLICIT_REF_COUNT_TO_USER=1)
      list (APPEND TEST_INVOCATION --ref-count-includes-implicit)
      string (APPEND TEST_NAME "-ReportImplicitRefCount")
    endif ()
    if (ARG_ALLOW_USING_RELEASED_OBJECTS)
      list (APPEND TEST_ENVIRONMENT ALLOW_USING_RELEASED_OBJECTS=1)
      list (APPEND TEST_INVOCATION --use-released-objects)
      string (APPEND TEST_NAME "-AllowUsingReleasedObjects")
    endif()
    if (ARG_ALLOW_USING_INACCESSIBLE_OBJECTS)
      list (APPEND TEST_ENVIRONMENT ALLOW_USING_INACCESSIBLE_OBJECTS=1)
      list (APPEND TEST_INVOCATION --use-inaccessible-objects)
      string (APPEND TEST_NAME "-AllowUsingInaccessibleObjects")
    endif()

    set (TEST_LOG "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.log")
    list (APPEND TEST_ENVIRONMENT CL_OBJECT_LIFETIME_LOG_FILENAME=${TEST_LOG})

    add_test (
        NAME ${TEST_NAME}
        COMMAND "${CMAKE_COMMAND}"
            "-DCOMMAND=${TEST_INVOCATION}"
            -DEXTRA_OUTPUT=${TEST_LOG}
            -DEXPECTED_EXTRA_OUTPUT=${ARG_REGEX}
            -P ${CMAKE_CURRENT_LIST_DIR}/object_lifetime_test.cmake
    )
    set_tests_properties (${TEST_NAME} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
endfunction()

add_layer_test_exe (TestBasicCounting     basic_counting.cpp)
add_layer_test_exe (TestLifetimeEdgeCases lifetime_edge_cases.cpp)
add_layer_test_exe (TestSubBuffer         sub_buffer.cpp)
add_layer_test_exe (TestPipeline          pipeline.cpp)
add_layer_test_exe (TestCreateImage       create_image.cpp)
add_layer_test_exe (TestSubDevice         sub_device.cpp)
add_layer_test_exe (TestCL200Calls        cl200_calls.cpp)
add_layer_test_exe (TestCloneKernel       clone_kernel.cpp)
add_layer_test_exe (TestCL300Calls        cl300_calls.cpp)
add_layer_test_exe (TestResurrect         resurrect.cpp)

add_layer_test (TestBasicCounting 200
  REGEX "${CMAKE_CURRENT_SOURCE_DIR}/basic_counting.regex"
)

# add_layer_test (TestLifetimeEdgeCases 200
#   REGEX "${CMAKE_CURRENT_SOURCE_DIR}/lifetime_edge_cases.regex")

add_layer_test (TestSubBuffer 200
  REGEX "${CMAKE_CURRENT_SOURCE_DIR}/sub_buffer.regex"
)

# add_layer_test (TestPipeline 200)

add_layer_test (TestCreateImage 200
  REGEX "${CMAKE_CURRENT_SOURCE_DIR}/create_image.regex"
)

# add_layer_test (TestSubDevice 200)
# add_layer_test (TestCL200Calls 200)

add_layer_test (TestCloneKernel 200
  REGEX "${CMAKE_CURRENT_SOURCE_DIR}/clone_kernel.regex"
)

# add_layer_test (TestCL300Calls 200)
# add_layer_test (TestResurrect 200)
