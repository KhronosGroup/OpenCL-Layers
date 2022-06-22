find_package(Threads REQUIRED)

function (add_layer_test_exe NAME SOURCE)
    add_executable (${NAME} ${SOURCE} object_lifetime_test.cpp)
    target_link_libraries (
        ${NAME}
        PRIVATE
            LayersCommon
            OpenCL::OpenCL
            Threads::Threads
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
    cmake_parse_arguments(PARSE_ARGV 2 ARG "TEST_IMPLICIT_REF_COUNT;TEST_ALLOW_INACCESSIBLE;TEST_ALLOW_RELEASED" "REGEX" "")

    set (TEST_ENVIRONMENT
        OPENCL_LAYERS=$<TARGET_FILE:CLObjectLifetimeLayer>
        OPENCL_OBJECT_LIFETIME_TRANSPARENT=0
        OPENCL_OBJECT_LIFETIME_LOG_SINK=file
        OCL_ICD_FILENAMES=$<TARGET_FILE:CLObjectLifetimeICD>
        OBJECT_LIFETIME_ICD_VERSION=${OPENCL_VERSION}
    )
    set (TEST_NAME "ObjectLifetime-${TEST_EXE}-CL${OPENCL_VERSION}")

    set (TEST_INVOCATION $<TARGET_FILE:${TEST_EXE}>)

    if (ARG_TEST_IMPLICIT_REF_COUNT)
      list (APPEND TEST_ENVIRONMENT REPORT_IMPLICIT_REF_COUNT_TO_USER=1)
      list (APPEND TEST_INVOCATION --ref-count-includes-implicit)
      string (APPEND TEST_NAME "-ImplicitRefCount")
    endif ()

    if (ARG_TEST_ALLOW_INACCESSIBLE)
      list (APPEND TEST_ENVIRONMENT ALLOW_USING_INACCESSIBLE_OBJECTS=1)
      string (APPEND TEST_NAME "-AllowInaccessible")
    endif ()

    if (ARG_TEST_ALLOW_RELEASED)
      list (APPEND TEST_ENVIRONMENT ALLOW_USING_RELEASED_OBJECTS=1)
      string (APPEND TEST_NAME "-AllowReleased")
    endif ()

    set (TEST_LOG "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.log")
    list (APPEND TEST_ENVIRONMENT OPENCL_OBJECT_LIFETIME_LOG_FILENAME=${TEST_LOG})

    add_test (
        NAME ${TEST_NAME}
        COMMAND "${CMAKE_COMMAND}"
            "-DCOMMAND=${TEST_INVOCATION}"
            "-DEXTRA_OUTPUT=${TEST_LOG}"
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
add_layer_test_exe (TestLeak              leak.cpp)
add_layer_test_exe (TestNonExistant       non_existant.cpp)
add_layer_test_exe (TestInvalidType       invalid_type.cpp)
add_layer_test_exe (TestBasicCountingPar  basic_counting_parallel.cpp)
add_layer_test_exe (TestCrossCountingPar  cross_counting_parallel.cpp)

function (add_layer_tests OPENCL_VERSION TEST_ARGS)
  if (OPENCL_VERSION EQUAL 200)
    set(SPECIFIC_REGEX_EXT "cl200.regex")
  else()
    set(SPECIFIC_REGEX_EXT "regex")
  endif()

  add_layer_test (TestBasicCounting ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/basic_counting.regex"
    ${TEST_ARGS}
  )
  add_layer_test (TestBasicCountingPar ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/basic_counting_parallel.regex"
    ${TEST_ARGS}
  )
  add_layer_test (TestCrossCountingPar ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/cross_counting_parallel.regex"
    ${TEST_ARGS}
  )
  add_layer_test (TestSubBuffer ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/sub_buffer.${SPECIFIC_REGEX_EXT}"
    ${TEST_ARGS}
  )
  add_layer_test (TestPipeline ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/pipeline.${SPECIFIC_REGEX_EXT}"
    ${TEST_ARGS}
  )
  add_layer_test (TestCreateImage ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/create_image.regex"
    ${TEST_ARGS}
  )
  add_layer_test (TestSubDevice ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/sub_device.${SPECIFIC_REGEX_EXT}"
    ${TEST_ARGS}
  )
  add_layer_test (TestLeak ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/leak.${SPECIFIC_REGEX_EXT}"
    ${TEST_ARGS}
  )
  add_layer_test (TestNonExistant ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_LIST_DIR}/non_existant.regex"
    ${TEST_ARGS}
  )
  add_layer_test (TestInvalidType ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_LIST_DIR}/invalid_type.regex"
    ${TEST_ARGS}
  )
  add_layer_test (TestLifetimeEdgeCases ${OPENCL_VERSION}
    REGEX "${CMAKE_CURRENT_SOURCE_DIR}/lifetime_edge_cases.${SPECIFIC_REGEX_EXT}"
    ${TEST_ARGS}
  )
  if(OPENCL_VERSION GREATER_EQUAL 200)
    add_layer_test (TestCL200Calls ${OPENCL_VERSION}
      REGEX "${CMAKE_CURRENT_SOURCE_DIR}/cl200_calls.regex"
      ${TEST_ARGS}
    )
  endif()
  if(OPENCL_VERSION GREATER_EQUAL 210)
    add_layer_test (TestCloneKernel ${OPENCL_VERSION}
      REGEX "${CMAKE_CURRENT_SOURCE_DIR}/clone_kernel.${SPECIFIC_REGEX_EXT}"
      ${TEST_ARGS}
    )
    add_layer_test (TestCL300Calls 300
      REGEX "${CMAKE_CURRENT_SOURCE_DIR}/cl300_calls.regex"
      ${TEST_ARGS}
    )
    add_layer_test (TestResurrect 300
      REGEX "${CMAKE_CURRENT_SOURCE_DIR}/resurrect.regex"
      ${TEST_ARGS} TEST_ALLOW_INACCESSIBLE
    )
  endif()
endfunction ()

# Note, opencl 1.x tests do not support reporting implicit ref count
add_layer_tests(120 "")
add_layer_tests(120 "TEST_ALLOW_RELEASED")

# Note, opencl 2.0 tests require allowing released objects as querying the refcount of
# objects which are being kept alive only by its children should return a refcount of 0
# and not return CL_INVALID.
add_layer_tests(200 "TEST_ALLOW_RELEASED")
add_layer_tests(200 "TEST_IMPLICIT_REF_COUNT;TEST_ALLOW_RELEASED")

add_layer_tests(300 "")
add_layer_tests(300 "TEST_IMPLICIT_REF_COUNT")
add_layer_tests(300 "TEST_ALLOW_RELEASED")
add_layer_tests(300 "TEST_IMPLICIT_REF_COUNT;TEST_ALLOW_RELEASED")
