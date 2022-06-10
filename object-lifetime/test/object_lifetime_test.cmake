execute_process(
    COMMAND ${COMMAND}
    OUTPUT_VARIABLE COMMAND_STDOUT
    RESULT_VARIABLE COMMAND_EXIT
)

message("${COMMAND_STDOUT}")

if(NOT COMMAND_EXIT STREQUAL 0)
    message(FATAL_ERROR "${COMMAND} exited with: ${COMMAND_EXIT}")
endif()

if(EXPECTED_OUTPUT)
    file(READ "${EXPECTED_OUTPUT}" EXPECTED)
    if(NOT "${COMMAND_STDOUT}" MATCHES "^${EXPECTED}$")
        message(FATAL_ERROR "Mismatch in output: ${COMMAND_STDOUT}, expected: ${EXPECTED}")
    endif()
endif()

if(EXTRA_OUTPUT)
    file(READ "${EXTRA_OUTPUT}"          OUTPUT_FILE)
    file(READ "${EXPECTED_EXTRA_OUTPUT}" EXPECTED_FILE)

    if(EXPECTED_FILE) # More friendly towards empty regexes
        if(NOT "${OUTPUT_FILE}" MATCHES "^${EXPECTED_FILE}$")
            message(FATAL_ERROR "Mismatch in output: ${OUTPUT_FILE}, expected: ${EXPECTED_FILE}")
        endif()
    endif()

    file(REMOVE "${EXTRA_OUTPUT}")
endif()
