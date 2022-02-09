file(READ "${EXPECTED_OUTPUT}" EXPECTED)

execute_process(
    COMMAND "${COMMAND}"
    OUTPUT_VARIABLE COMMAND_STDOUT
    RESULT_VARIABLE COMMAND_EXIT)

message("${COMMAND_STDOUT}")

if(NOT COMMAND_EXIT STREQUAL 0)
    message(FATAL_ERROR "${COMMAND} exited with: ${COMMAND_EXIT}")
endif()

if(NOT "${COMMAND_STDOUT}" MATCHES "${EXPECTED}")
    message(FATAL_ERROR "Mismatch in output: ${COMMAND_STDOUT}, expected: ${EXPECTED}")
endif()

if(DEFINED CMAKE_ARGV4)
    file(READ "${EXTRA_OUTPUT}"          OUTPUT_FILE)
    file(READ "${EXPECTED_EXTRA_OUTPUT}" EXPECTED_FILE)

    if(NOT "${OUTPUT_FILE}" MATCHES "${EXPECTED_FILE}")
        message(FATAL_ERROR "Mismatch in output: ${OUTPUT_FILE}, expected: ${EXPECTED_FILE}")
    endif()

    file(REMOVE "${EXTRA_OUTPUT}")
endif()
