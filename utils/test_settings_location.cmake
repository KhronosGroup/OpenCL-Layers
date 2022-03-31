include(CMakeParseArguments)

set(DIR_WITH_EXISTING_DEFAULT_NAME_FILE1 "${CMAKE_CURRENT_BINARY_DIR}/default")
set(PATH_TO_EXISTING_DEFAULT_NAME_FILE1  "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}/cl_layer_settings.txt")
set(DIR_WITH_EXISTING_DEFAULT_NAME_FILE2 "${CMAKE_CURRENT_BINARY_DIR}/default2")
set(PATH_TO_EXISTING_DEFAULT_NAME_FILE2  "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE2}/cl_layer_settings.txt")
set(PATH_TO_EXISTING_CUSTOM_NAME_FILE1   "${CMAKE_CURRENT_BINARY_DIR}/custom/cl_custom_settings.txt")
set(PATH_TO_MISSING_CUSTOM_NAME_FILE     "${CMAKE_CURRENT_BINARY_DIR}/missing.txt")

file(WRITE "${PATH_TO_EXISTING_DEFAULT_NAME_FILE1}" " ")
file(WRITE "${PATH_TO_EXISTING_DEFAULT_NAME_FILE2}" " ")
file(WRITE "${PATH_TO_EXISTING_CUSTOM_NAME_FILE1}" " ")

function(test_settings_location)
  set(FLAGS SETTINGS_IN_HOME)
  set(ONE_VALUE_KEYWORDS NAME EXPECTED XDG_DATA_HOME SETTINGS_ENV SETTINGS_REG REG_TAG)
  set(MULTI_VALUE_KEYWORDS ENVIRONMENT)
  cmake_parse_arguments(TSL "${FLAGS}" "${ONE_VALUE_KEYWORDS}" "${MULTI_VALUE_KEYWORDS}" "${ARGV}")

  if(TSL_SETTINGS_ENV)
    set(SETTINGS_ENV_ARG "OPENCL_LAYERS_SETTINGS_PATH=${TSL_SETTINGS_ENV}")
  else()
    set(SETTINGS_ENV_ARG "--unset=OPENCL_LAYERS_SETTINGS_PATH")
  endif()

  if(NOT WIN32)
    if(TSL_XDG_DATA_HOME)
      set(XDG_DATA_HOME_ARG "XDG_DATA_HOME=${TSL_XDG_DATA_HOME}")
    else()
      set(XDG_DATA_HOME_ARG "--unset=XDG_DATA_HOME")
    endif()

    if(TSL_SETTINGS_IN_HOME)
      set(HOME_ARG "HOME=${CMAKE_CURRENT_BINARY_DIR}/home/has_settings")
    else()
      set(HOME_ARG "HOME=${CMAKE_CURRENT_BINARY_DIR}/home/no_settings")
    endif()
  endif()

  add_test(
    NAME "${TSL_NAME}"
    COMMAND
      ${CMAKE_COMMAND} -E env ${XDG_DATA_HOME_ARG} ${HOME_ARG} ${SETTINGS_ENV_ARG}
      $<TARGET_FILE:print_settings_location>)

  if(WIN32)
    # Get-Item HKCU:\SOFTWARE\Khronos\OpenCL\Settings | Select-Object -ExpandProperty Property | Out-File C:\Users\mate\Desktop\Settings.txt
    # Get-Item HKCU:\SOFTWARE\Khronos\OpenCL\Settings | Select-Object -ExpandProperty Property | ForEach-Object { Remove-ItemProperty -Path HKCU:\SOFTWARE\Khronos\OpenCL\Settings -Name $_ }
    # Get-Content C:\Users\mate\Desktop\Settings.txt | ForEach-Object { New-ItemProperty -Path HKCU:\SOFTWARE\Khronos\OpenCL\Settings -Name $_  -Value "0" | Out-Null }
    set(SETTINGS_REG_PATH "HKCU:/Software/Khronos/OpenCL/Settings")
    set(SETTINGS_REG_VALUE_CACHE "${CMAKE_CURRENT_BINARY_DIR}/UserRegistry.txt")
    string(RANDOM LENGTH 4 FITXTURE_TAG)
    add_test(
      NAME Backup-LayerSettings-${FITXTURE_TAG}
      COMMAND powershell.exe -Command "& { if (Test-Path ${SETTINGS_REG_PATH}) { Get-Item ${SETTINGS_REG_PATH} | Select-Object -ExpandProperty Property | Out-File ${SETTINGS_REG_VALUE_CACHE} -Encoding ascii } }"
    )
    set_tests_properties(Backup-LayerSettings-${FITXTURE_TAG}
      PROPERTIES
        FIXTURES_SETUP Fixture-${FITXTURE_TAG}
        RESOURCE_LOCK Registry
    )
    add_test(
      NAME Initialize-LayerSettings-${FITXTURE_TAG}
      COMMAND powershell.exe -Command "& { if (Test-Path ${SETTINGS_REG_PATH}) { Get-Item ${SETTINGS_REG_PATH} | Select-Object -ExpandProperty Property | ForEach-Object { Remove-ItemProperty -Path ${SETTINGS_REG_PATH} -Name $_  } } else { New-Item ${SETTINGS_REG_PATH} -Force | Out-Null } }"
    )
    set_tests_properties(Initialize-LayerSettings-${FITXTURE_TAG}
      PROPERTIES
        FIXTURES_SETUP Fixture-${FITXTURE_TAG}
        RESOURCE_LOCK Registry
        DEPENDS Backup-LayerSettings-${FITXTURE_TAG}
    )
    if(TSL_SETTINGS_REG)
      add_test(
        NAME Register-LayerSettings-${FITXTURE_TAG}
        COMMAND powershell.exe -Command "& { New-ItemProperty -Type DWORD -Path ${SETTINGS_REG_PATH} -Name '${TSL_SETTINGS_REG}'.replace('`n','').replace('`r','') -Value 0 }"
      )
      set_tests_properties(Register-LayerSettings-${FITXTURE_TAG}
        PROPERTIES
          FIXTURES_SETUP Fixture-${FITXTURE_TAG}
          RESOURCE_LOCK Registry
          DEPENDS Initialize-LayerSettings-${FITXTURE_TAG}
      )
    endif()
    add_test(
      NAME Restore-LayerSettings-${FITXTURE_TAG}
      COMMAND powershell.exe -Command "& { if (Test-Path ${SETTINGS_REG_VALUE_CACHE}) { Get-Content ${SETTINGS_REG_VALUE_CACHE} | ForEach-Object { New-ItemProperty -Path ${SETTINGS_REG_PATH} -Name $_.replace('`n','').replace('`r','') -Value 0 | Out-Null } } else { Remove-Item ${SETTINGS_REG_PATH} } }"
    )
    set_tests_properties(Restore-LayerSettings-${FITXTURE_TAG}
      PROPERTIES
        FIXTURES_CLEANUP Fixture-${FITXTURE_TAG}
        RESOURCE_LOCK Registry
    )

    set_tests_properties("${TSL_NAME}"
      PROPERTIES
        FIXTURES_REQUIRED Fixture-${FITXTURE_TAG}
        RESOURCE_LOCK Registry
        PASS_REGULAR_EXPRESSION "${TSL_EXPECTED}"
    )
  endif()
endfunction()

# When no relevant environment variables or registry values are set, layers
# should default to looking for settings in the current working directory
test_settings_location(
  NAME SettingsLocation-Default
  EXPECTED "cl_layer_settings.txt"
)

if(WIN32)
  # When OPENCL_LAYERS_SETTINGS_PATH env var is set, it should override every
  # other condition.
  test_settings_location(
    NAME SettingsLocation-EnvFile
    EXPECTED     "${PATH_TO_EXISTING_CUSTOM_NAME_FILE1}"
    SETTINGS_ENV "${PATH_TO_EXISTING_CUSTOM_NAME_FILE1}"
    SETTINGS_REG "${PATH_TO_EXISTING_DEFAULT_NAME_FILE2}"
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE2}"
  )
  test_settings_location(
    NAME SettingsLocation-EnvDir
    EXPECTED     "${PATH_TO_EXISTING_DEFAULT_NAME_FILE1}"
    SETTINGS_ENV "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
    SETTINGS_REG "${PATH_TO_EXISTING_DEFAULT_NAME_FILE2}"
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE2}"
  )
  # When OPENCL_LAYERS_SETTINGS_PATH env var is set, but points to a non-existing
  # path, it should fall back to platform-specific paths
  test_settings_location(
    NAME SettingsLocation-EnvMissingFile-RegFile
    EXPECTED     "${PATH_TO_EXISTING_CUSTOM_NAME_FILE1}"
    SETTINGS_ENV "${PATH_TO_MISSING_CUSTOM_NAME_FILE}"
    SETTINGS_REG "${PATH_TO_EXISTING_CUSTOM_NAME_FILE1}"
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
  )
  # When OPENCL_LAYERS_SETTINGS_PATH env var is set, but points to a non-existing
  # path, registry holds an entry which doesn't exist, it should fall back to cwd
  test_settings_location(
    NAME SettingsLocation-EnvMissingFile-RegMissingFile
    EXPECTED     "cl_layer_settings.txt"
    SETTINGS_ENV "${PATH_TO_MISSING_CUSTOM_NAME_FILE}"
    SETTINGS_REG "${PATH_TO_MISSING_CUSTOM_NAME_FILE}"
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
  )
endif()

##############################################################################

#function(run_cmd_expect)
#    cmake_parse_arguments(ARG "" "EXPECTED" "COMMAND" "${ARGV}")
#
#    execute_process(
#        COMMAND ${ARG_COMMAND}
#        OUTPUT_VARIABLE COMMAND_STDOUT
#        RESULT_VARIABLE COMMAND_EXIT)
#
#    if(NOT COMMAND_EXIT STREQUAL 0)
#        message(FATAL_ERROR "${TARGET_FILE} exited with: ${COMMAND_EXIT}")
#    endif()
#
#    if(NOT COMMAND_STDOUT STREQUAL ARG_EXPECTED)
#        message(FATAL_ERROR "${TARGET_FILE} has output: ${COMMAND_STDOUT}, expected: ${ARG_EXPECTED}")
#    endif()
#endfunction()
#
## Clean up the folders if the last test run failed or didn't clean them up for whatever reason
#file(REMOVE_RECURSE ${BINARY_DIR}/.test_data)
#
## No relevant environment variables are set, should look the file in the current directory
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=XDG_DATA_HOME --unset=HOME ${TARGET_FILE}
#               EXPECTED "cl_layer_settings.txt")
#
#set(XDG_DATA_HOME                    ${BINARY_DIR}/.test_data/xdg_data_home)
#set(HOME                             ${BINARY_DIR}/.test_data/home)
#set(OPENCL_LAYERS_SETTINGS_PATH      ${BINARY_DIR}/.test_data/explicit_folder)
#set(OPENCL_LAYERS_SETTINGS_PATH_FILE ${OPENCL_LAYERS_SETTINGS_PATH}/explicit.txt)
#
## If $XDG_DATA_HOME is set but no opencl/settings.d/cl_layer_settings.txt exists in it fall back
## to the current working directory
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=HOME --unset=OPENCL_LAYERS_SETTINGS_PATH
#                   XDG_DATA_HOME=${XDG_DATA_HOME} ${TARGET_FILE}
#               EXPECTED "cl_layer_settings.txt")
#
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=XDG_DATA_HOME --unset=OPENCL_LAYERS_SETTINGS_PATH
#                   HOME=${XDG_DATA_HOME} ${TARGET_FILE}
#               EXPECTED "cl_layer_settings.txt")
#
## Create cl_layer_settings.txt in the binary directory for all tests
#file(WRITE ${XDG_DATA_HOME}/opencl/settings.d/cl_layer_settings.txt "")
#file(WRITE ${HOME}/.local/share/opencl/settings.d/cl_layer_settings.txt "")
#file(WRITE ${OPENCL_LAYERS_SETTINGS_PATH}/cl_layer_settings.txt "")
#file(WRITE ${OPENCL_LAYERS_SETTINGS_PATH_FILE} "")
#
## If $XDG_DATA_HOME is set look for opencl/settings.d/cl_layer_settings.txt in it
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=HOME --unset=OPENCL_LAYERS_SETTINGS_PATH
#                   XDG_DATA_HOME=${XDG_DATA_HOME} ${TARGET_FILE}
#               EXPECTED "${XDG_DATA_HOME}/opencl/settings.d/cl_layer_settings.txt")
#
## Otherwise if $HOME is set look for $HOME/.local/share/opencl/settings.d/cl_layer_settings.txt in it
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=XDG_DATA_HOME --unset=OPENCL_LAYERS_SETTINGS_PATH
#                   HOME=${HOME} ${TARGET_FILE}
#               EXPECTED "${HOME}/.local/share/opencl/settings.d/cl_layer_settings.txt")
#
## XDG_DATA_HOME should have higher precedence than $HOME
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=OPENCL_LAYERS_SETTINGS_PATH
#                   HOME=${HOME} XDG_DATA_HOME=${XDG_DATA_HOME} ${TARGET_FILE}
#               EXPECTED "${XDG_DATA_HOME}/opencl/settings.d/cl_layer_settings.txt")
#
## if $OPENCL_LAYERS_SETTINGS_PATH is set and is a directory look for the file cl_layer_settings.txt in it
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=HOME --unset=XDG_DATA_HOME
#                   OPENCL_LAYERS_SETTINGS_PATH=${OPENCL_LAYERS_SETTINGS_PATH} ${TARGET_FILE}
#               EXPECTED "${OPENCL_LAYERS_SETTINGS_PATH}/cl_layer_settings.txt")
#
## if $OPENCL_LAYERS_SETTINGS_PATH is set and is a file use it
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env --unset=HOME --unset=XDG_DATA_HOME
#                   OPENCL_LAYERS_SETTINGS_PATH=${OPENCL_LAYERS_SETTINGS_PATH_FILE} ${TARGET_FILE}
#               EXPECTED "${OPENCL_LAYERS_SETTINGS_PATH_FILE}")
#
## $OPENCL_LAYERS_SETTINGS_PATH should have a priority over $XDG_DATA_HOME and $HOME
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env HOME=${HOME} --unset=XDG_DATA_HOME
#                   OPENCL_LAYERS_SETTINGS_PATH=${OPENCL_LAYERS_SETTINGS_PATH_FILE} ${TARGET_FILE}
#               EXPECTED "${OPENCL_LAYERS_SETTINGS_PATH_FILE}")
#run_cmd_expect(COMMAND ${CMAKE_COMMAND} -E env HOME=${HOME} XDG_DATA_HOME=${XDG_DATA_HOME}
#                   OPENCL_LAYERS_SETTINGS_PATH=${OPENCL_LAYERS_SETTINGS_PATH_FILE} ${TARGET_FILE}
#               EXPECTED "${OPENCL_LAYERS_SETTINGS_PATH_FILE}")
#
## Clean up the files
#file(REMOVE_RECURSE ${BINARY_DIR}/.test_data)
#endif()