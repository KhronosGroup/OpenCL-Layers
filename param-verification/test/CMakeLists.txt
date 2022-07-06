function (add_param_verification_test_exe NAME SOURCE)
    add_executable (${NAME} ${SOURCE} param_verification_test.cpp param_verification_test.hpp)
    target_link_libraries (
        ${NAME}
        PRIVATE
            LayersTest
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
endfunction ()

function (add_param_verification_test TEST_EXE OPENCL_VERSION)
    cmake_parse_arguments(PARSE_ARGV 2 ARG "ENABLE_OBJECT_LIFETIME_LAYER" "REGEX" "")

    if (ARG_ENABLE_OBJECT_LIFETIME_LAYER)
      if (WIN32)
        set (TEST_ENVIRONMENT
            "OPENCL_LAYERS=$<TARGET_FILE:CLObjectLifetimeLayer>\;$<TARGET_FILE:CLParamVerificationLayer>"
        )
      else ()
        set (TEST_ENVIRONMENT
            OPENCL_LAYERS=$<TARGET_FILE:CLObjectLifetimeLayer>:$<TARGET_FILE:CLParamVerificationLayer>
        )
      endif ()
      list (APPEND TEST_ENVIRONMENT OPENCL_OBJECT_LIFETIME_LOG_SINK=file)
    else ()
      set (TEST_ENVIRONMENT
          OPENCL_LAYERS=$<TARGET_FILE:CLParamVerificationLayer>
      )
    endif ()

    list (APPEND TEST_ENVIRONMENT
        OPENCL_PARAM_VERIFICATION_LOG_SINK=file
        OCL_ICD_FILENAMES=$<TARGET_FILE:CLObjectLifetimeICD>
        OBJECT_LIFETIME_ICD_VERSION=${OPENCL_VERSION}
        ALWAYS_RETURN_SUCCESS=1
    )
    set (TEST_NAME "ParamVerification-${TEST_EXE}-CL${OPENCL_VERSION}")

    set (TEST_INVOCATION $<TARGET_FILE:${TEST_EXE}>)

    set (TEST_LOG "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}.log")
    list (APPEND TEST_ENVIRONMENT OPENCL_PARAM_VERIFICATION_LOG_FILENAME=${TEST_LOG})

    if (ARG_ENABLE_OBJECT_LIFETIME_LAYER)
    set (LIFETIME_TEST_LOG "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}-Lifetime.log")
      list (APPEND TEST_ENVIRONMENT OPENCL_OBJECT_LIFETIME_LOG_FILENAME=${LIFETIME_TEST_LOG})
    endif ()

    add_test (
        NAME ${TEST_NAME}
        COMMAND "${CMAKE_COMMAND}"
            "-DCOMMAND=${TEST_INVOCATION}"
            -DEXTRA_OUTPUT=${TEST_LOG}
            -DEXPECTED_EXTRA_OUTPUT=${ARG_REGEX}
            -P ${PROJECT_SOURCE_DIR}/cmake/run_and_compare.cmake
    )
    set_tests_properties (${TEST_NAME} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
endfunction ()

add_param_verification_test_exe (TestBasic          basic.cpp)
add_param_verification_test_exe (TestPointers       pointers.cpp)
add_param_verification_test_exe (TestLimits         limits.cpp)
add_param_verification_test_exe (TestFlags          flags.cpp)
add_param_verification_test_exe (TestBounds         bounds.cpp)
add_param_verification_test_exe (TestContextSharing context_sharing.cpp)
add_param_verification_test_exe (TestStructs        structs.cpp)
add_param_verification_test_exe (TestProperties     properties.cpp)
add_param_verification_test_exe (TestObjectValidity object_validity.cpp)

foreach (VERSION 120 200 300)
    add_param_verification_test (TestBasic          ${VERSION} REGEX ${CMAKE_CURRENT_SOURCE_DIR}/basic.regex)
    add_param_verification_test (TestPointers       ${VERSION} REGEX ${CMAKE_CURRENT_SOURCE_DIR}/pointers.regex)
    add_param_verification_test (TestLimits         ${VERSION} REGEX ${CMAKE_CURRENT_SOURCE_DIR}/limits.regex)
    add_param_verification_test (TestFlags          ${VERSION} REGEX ${CMAKE_CURRENT_SOURCE_DIR}/flags.regex)
    add_param_verification_test (TestBounds         ${VERSION} REGEX ${CMAKE_CURRENT_SOURCE_DIR}/bounds.regex)
    add_param_verification_test (TestContextSharing ${VERSION} REGEX ${CMAKE_CURRENT_SOURCE_DIR}/context_sharing.regex)
    if (${VERSION} GREATER_EQUAL 200)
        add_param_verification_test (TestProperties ${VERSION} REGEX ${CMAKE_CURRENT_SOURCE_DIR}/properties.regex)
    endif ()
    add_param_verification_test (TestObjectValidity ${VERSION} REGEX ${CMAKE_CURRENT_SOURCE_DIR}/object_validity.regex ENABLE_OBJECT_LIFETIME_LAYER)
endforeach ()
