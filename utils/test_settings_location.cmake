function(run_cmd_expect)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "EXPECTED" "COMMAND")

    execute_process(
        COMMAND ${ARG_COMMAND}
        OUTPUT_VARIABLE COMMAND_STDOUT
        RESULT_VARIABLE COMMAND_EXIT)

    if(NOT COMMAND_EXIT STREQUAL 0)
        message(FATAL_ERROR "${TARGET_FILE} exited with: ${COMMAND_EXIT}")
    endif()

    if(NOT COMMAND_STDOUT STREQUAL ARG_EXPECTED)
        message(FATAL_ERROR "${TARGET_FILE} has output: ${COMMAND_STDOUT}, expected: ${ARG_EXPECTED}")
    endif()
endfunction()

# Clean up the folders if the last test run failed or didn't clean them up for whatever reason
file(REMOVE_RECURSE ${BINARY_DIR}/.test_data)

# No relevant environment variables are set, should look the file in the current directory
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=XDG_DATA_HOME --unset=HOME ${TARGET_FILE}
               EXPECTED "cl_layer_settings.txt")

set(XDG_DATA_HOME                    ${BINARY_DIR}/.test_data/xdg_data_home)
set(HOME                             ${BINARY_DIR}/.test_data/home)
set(OPENCL_LAYERS_SETTINGS_PATH      ${BINARY_DIR}/.test_data/explicit_folder)
set(OPENCL_LAYERS_SETTINGS_PATH_FILE ${OPENCL_LAYERS_SETTINGS_PATH}/explicit.txt)

# If $XDG_DATA_HOME is set but no opencl/settings.d/cl_layer_settings.txt exists in it fall back
# to the current working directory
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=HOME --unset=OPENCL_LAYERS_SETTINGS_PATH
                   XDG_DATA_HOME=${XDG_DATA_HOME} ${TARGET_FILE}
               EXPECTED "cl_layer_settings.txt")

run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=XDG_DATA_HOME --unset=OPENCL_LAYERS_SETTINGS_PATH
                   HOME=${XDG_DATA_HOME} ${TARGET_FILE}
               EXPECTED "cl_layer_settings.txt")

# Create cl_layer_settings.txt in the binary directory for all tests
file(WRITE ${XDG_DATA_HOME}/opencl/settings.d/cl_layer_settings.txt "")
file(WRITE ${HOME}/.local/share/opencl/settings.d/cl_layer_settings.txt "")
file(WRITE ${OPENCL_LAYERS_SETTINGS_PATH}/cl_layer_settings.txt "")
file(WRITE ${OPENCL_LAYERS_SETTINGS_PATH_FILE} "")

# If $XDG_DATA_HOME is set look for opencl/settings.d/cl_layer_settings.txt in it
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=HOME --unset=OPENCL_LAYERS_SETTINGS_PATH
                   XDG_DATA_HOME=${XDG_DATA_HOME} ${TARGET_FILE}
               EXPECTED "${XDG_DATA_HOME}/opencl/settings.d/cl_layer_settings.txt")

# Otherwise if $HOME is set look for $HOME/.local/share/opencl/settings.d/cl_layer_settings.txt in it
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=XDG_DATA_HOME --unset=OPENCL_LAYERS_SETTINGS_PATH
                   HOME=${HOME} ${TARGET_FILE}
               EXPECTED "${HOME}/.local/share/opencl/settings.d/cl_layer_settings.txt")

# XDG_DATA_HOME should have higher precedence than $HOME
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=OPENCL_LAYERS_SETTINGS_PATH
                   HOME=${HOME} XDG_DATA_HOME=${XDG_DATA_HOME} ${TARGET_FILE}
               EXPECTED "${XDG_DATA_HOME}/opencl/settings.d/cl_layer_settings.txt")

# if $OPENCL_LAYERS_SETTINGS_PATH is set and is a directory look for the file cl_layer_settings.txt in it
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=HOME --unset=XDG_DATA_HOME
                   OPENCL_LAYERS_SETTINGS_PATH=${OPENCL_LAYERS_SETTINGS_PATH} ${TARGET_FILE}
               EXPECTED "${OPENCL_LAYERS_SETTINGS_PATH}/cl_layer_settings.txt")

# if $OPENCL_LAYERS_SETTINGS_PATH is set and is a file use it
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=HOME --unset=XDG_DATA_HOME
                   OPENCL_LAYERS_SETTINGS_PATH=${OPENCL_LAYERS_SETTINGS_PATH_FILE} ${TARGET_FILE}
               EXPECTED "${OPENCL_LAYERS_SETTINGS_PATH_FILE}")

# $OPENCL_LAYERS_SETTINGS_PATH should have a priority over $XDG_DATA_HOME and $HOME
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env HOME=${HOME} --unset=XDG_DATA_HOME
                   OPENCL_LAYERS_SETTINGS_PATH=${OPENCL_LAYERS_SETTINGS_PATH_FILE} ${TARGET_FILE}
               EXPECTED "${OPENCL_LAYERS_SETTINGS_PATH_FILE}")
run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env HOME=${HOME} XDG_DATA_HOME=${XDG_DATA_HOME}
                   OPENCL_LAYERS_SETTINGS_PATH=${OPENCL_LAYERS_SETTINGS_PATH_FILE} ${TARGET_FILE}
               EXPECTED "${OPENCL_LAYERS_SETTINGS_PATH_FILE}")

# Clean up the files
file(REMOVE_RECURSE ${BINARY_DIR}/.test_data)
