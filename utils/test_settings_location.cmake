include(CMakeParseArguments)

set(DIR_WITH_EXISTING_DEFAULT_NAME_FILE1 "${CMAKE_CURRENT_BINARY_DIR}/default")
set(PATH_TO_EXISTING_DEFAULT_NAME_FILE1  "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}/cl_layer_settings.txt")
set(DIR_WITH_EXISTING_DEFAULT_NAME_FILE2 "${CMAKE_CURRENT_BINARY_DIR}/default2")
set(PATH_TO_EXISTING_DEFAULT_NAME_FILE2  "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE2}/cl_layer_settings.txt")
set(PATH_TO_EXISTING_CUSTOM_NAME_FILE1   "${CMAKE_CURRENT_BINARY_DIR}/custom/cl_custom_settings.txt")
set(PATH_TO_MISSING_CUSTOM_NAME_FILE     "${CMAKE_CURRENT_BINARY_DIR}/missing.txt")
set(XDG_DIR_WITH_EXISTING_FILE           "${CMAKE_CURRENT_BINARY_DIR}/xdg_with")
set(XDG_DIR_WITH_MISSING_FILE            "${CMAKE_CURRENT_BINARY_DIR}/xdg_without")
set(XDG_PATH_TO_EXISTING_FILE            "${XDG_DIR_WITH_EXISTING_FILE}/settings.d/opencl/cl_layer_settings.txt")
set(HOME_DIR_WITH_EXISTING_FILE          "${CMAKE_CURRENT_BINARY_DIR}/home_with")
set(HOME_DIR_WITH_MISSING_FILE           "${CMAKE_CURRENT_BINARY_DIR}/home_without")
set(HOME_PATH_TO_EXISTING_FILE           "${HOME_DIR_WITH_EXISTING_FILE}/.local/share/cl_layer_settings.txt")

file(WRITE "${PATH_TO_EXISTING_DEFAULT_NAME_FILE1}" " ")
file(WRITE "${PATH_TO_EXISTING_DEFAULT_NAME_FILE2}" " ")
file(WRITE "${PATH_TO_EXISTING_CUSTOM_NAME_FILE1}" " ")
file(WRITE "${XDG_PATH_TO_EXISTING_FILE}" " ")
file(WRITE "${HOME_PATH_TO_EXISTING_FILE}" " ")

if(WIN32)
  # NOTE: The ICD loader will only inspect user registry (and environment variables) when
  #       running as normal user. It is decided at configuration time whether tests will
  #       write to system registry Hive Key Local Machine (admin, aka. high-integrity) or to
  #       Hive Key Current User. Configuring as user but running tests as admin won't work.
  execute_process(COMMAND
    powershell.exe "-Command" "& {(new-object System.Security.Principal.WindowsPrincipal([System.Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator).ToString().ToUpper()}"
    OUTPUT_VARIABLE HIGH_INTEGRITY_CHECK_OUTPUT
    RESULT_VARIABLE HIGH_INTEGRITY_CHECK_RESULT
  )
  if(NOT ${HIGH_INTEGRITY_CHECK_RESULT} EQUAL 0)
    message(FATAL_ERROR "Failed to detect presence of admin priviliges.")
  endif()
  string(STRIP "${HIGH_INTEGRITY_CHECK_OUTPUT}" HIGH_INTEGRITY_CHECK_OUTPUT) # Strip newline
  if(HIGH_INTEGRITY_CHECK_OUTPUT) # PowerShell "True"/"False" output coincides with CMake boolean
    set(HIVE HKLM)
  else()
    set(HIVE HKCU)
  endif()

  # NOTE: This neutralizes a footgun in Windows, namely how SysWOW64 and the registry works.
  #       When a 64-bit process (powershell) write a DWORD to the registry it becomes
  #       invisible to 32-bit processes. This will likely come as a surprise. When compiling
  #       32-bit executables, we must make sure to use a 32-bit powershell.exe
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(POWERSHELL_BIN "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe")
  else()
    set(POWERSHELL_BIN "C:\\Windows\\SysWOW64\\WindowsPowerShell\\v1.0\\powershell.exe")
  endif()
endif()

function(test_settings_location)
  set(FLAGS SETTINGS_HOME)
  set(ONE_VALUE_KEYWORDS NAME EXPECTED SETTINGS_XDG SETTINGS_ENV SETTINGS_REG REG_TAG)
  set(MULTI_VALUE_KEYWORDS ENVIRONMENT)
  cmake_parse_arguments(TSL "${FLAGS}" "${ONE_VALUE_KEYWORDS}" "${MULTI_VALUE_KEYWORDS}" "${ARGV}")

  if(TSL_SETTINGS_ENV)
    set(SETTINGS_ENV_ARG "OPENCL_LAYERS_SETTINGS_PATH=${TSL_SETTINGS_ENV}")
  else()
    set(SETTINGS_ENV_ARG "--unset=OPENCL_LAYERS_SETTINGS_PATH")
  endif()

  if(NOT WIN32)
    if(TSL_SETTINGS_XDG)
      set(XDG_DATA_HOME_ARG "XDG_DATA_HOME=${TSL_SETTINGS_XDG}")
    else()
      set(XDG_DATA_HOME_ARG "--unset=XDG_DATA_HOME")
    endif()

    if(TSL_SETTINGS_HOME)
      set(HOME_ARG "HOME=${HOME_DIR_WITH_EXISTING_FILE}")
    else()
      set(HOME_ARG "HOME=${HOME_DIR_WITH_MISSING_FILE}")
    endif()
  endif()

  if(NOT WIN32)
    add_test(
      NAME "${TSL_NAME}"
      COMMAND
        "${CMAKE_COMMAND}" -E env ${XDG_DATA_HOME_ARG} ${HOME_ARG} ${SETTINGS_ENV_ARG} $<TARGET_FILE:print_settings_location>
    )
  else()
    # Save Settings registry properties to file SETTINGS_REG_VALUE_CACHE
    # Remove all existing Settings properties
    # If user provided a reg entry, set it
    # Run test
    # Remove the user provided reg entry
    # Restore previous registry contents
    # Remove SETTINGS_REG_VALUE_CACHE file
    set(SETTINGS_REG_PATH "${HIVE}:/Software/Khronos/OpenCL/Settings")
    set(SETTINGS_REG_VALUE_CACHE "${CMAKE_CURRENT_BINARY_DIR}/UserRegistry.txt")
    add_test(
      NAME "${TSL_NAME}"
      COMMAND
      ${POWERSHELL_BIN} -Command "& { \
        if (Test-Path ${SETTINGS_REG_PATH}) \
        { \
          Get-Item ${SETTINGS_REG_PATH} | Select-Object -ExpandProperty Property | Out-File ${SETTINGS_REG_VALUE_CACHE} -Encoding ascii \
        }; \
        if (Test-Path ${SETTINGS_REG_PATH}) \
        { \
          Get-Item ${SETTINGS_REG_PATH} | Select-Object -ExpandProperty Property | ForEach-Object { Remove-ItemProperty -Path ${SETTINGS_REG_PATH} -Name $_  } \
        } else \
        { \
          New-Item ${SETTINGS_REG_PATH} -Force | Out-Null \
        }; \
        if ('${TSL_SETTINGS_REG}'.Length -ne 0) \
        { \
          New-ItemProperty -Type DWORD -Path ${SETTINGS_REG_PATH} -Name '${TSL_SETTINGS_REG}'.replace('`n','').replace('`r','') -Value 0 | Out-Null; \
          Get-ChildItem ${SETTINGS_REG_PATH}; \
        } \
        & \"${CMAKE_COMMAND}\" -E env ${XDG_DATA_HOME_ARG} ${HOME_ARG} ${SETTINGS_ENV_ARG} $<TARGET_FILE:print_settings_location> ;\
        if (Test-Path ${SETTINGS_REG_VALUE_CACHE}) \
        { \
          Remove-ItemProperty -Path ${SETTINGS_REG_PATH} -Name '${TSL_SETTINGS_REG}'.replace('`n','').replace('`r',''); \
          Get-Content ${SETTINGS_REG_VALUE_CACHE} | ForEach-Object { New-ItemProperty -Path ${SETTINGS_REG_PATH} -Name $_.replace('`n','').replace('`r','') -Value 0 | Out-Null } \
          Remove-Item ${SETTINGS_REG_VALUE_CACHE};
        } else \
        { \
          Remove-Item ${SETTINGS_REG_PATH} \
        };
      }"
    )
    set_tests_properties("${TSL_NAME}"
      PROPERTIES
        RESOURCE_LOCK Registry
    )
  endif()
  set_tests_properties("${TSL_NAME}"
      PROPERTIES
        PASS_REGULAR_EXPRESSION "^${TSL_EXPECTED}"
    )
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
elseif(UNIX)
  # When OPENCL_LAYERS_SETTINGS_PATH env var is set, it should override every
  # other condition.
  test_settings_location(
    NAME SettingsLocation-EnvFile-Xdg-Home
    EXPECTED     "${PATH_TO_EXISTING_CUSTOM_NAME_FILE1}"
    SETTINGS_ENV "${PATH_TO_EXISTING_CUSTOM_NAME_FILE1}"
    SETTINGS_XDG "${XDG_DIR_WITH_EXISTING_FILE}"
    SETTINGS_HOME
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE2}"
  )
  test_settings_location(
    NAME SettingsLocation-EnvDir-Xdg-Home
    EXPECTED     "${PATH_TO_EXISTING_DEFAULT_NAME_FILE1}"
    SETTINGS_ENV "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
    SETTINGS_XDG "${XDG_DIR_WITH_EXISTING_FILE}"
    SETTINGS_HOME
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE2}"
  )
  # When OPENCL_LAYERS_SETTINGS_PATH env var is set, but points to a non-existing
  # path, it should fall back to platform-specific paths
  test_settings_location(
    NAME SettingsLocation-EnvMissingFile-Xdg
    EXPECTED     "${XDG_PATH_TO_EXISTING_FILE}"
    SETTINGS_ENV "${PATH_TO_MISSING_CUSTOM_NAME_FILE}"
    SETTINGS_XDG "${XDG_DIR_WITH_EXISTING_FILE}"
    SETTINGS_HOME
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
  )
  test_settings_location(
    NAME SettingsLocation-EnvMissingFile-XdgMissing-Home
    EXPECTED     "cl_layer_settings.txt"
    SETTINGS_ENV "${PATH_TO_MISSING_CUSTOM_NAME_FILE}"
    SETTINGS_XDG "${XDG_DIR_WITH_MISSING_FILE}"
    SETTINGS_HOME
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
  )
  # When OPENCL_LAYERS_SETTINGS_PATH env var is set, but points to a non-existing
  # path, registry holds an entry which doesn't exist, it should fall back to cwd
  test_settings_location(
    NAME SettingsLocation-EnvMissingFile-XdgMissing-HomeMissing
    EXPECTED     "cl_layer_settings.txt"
    SETTINGS_ENV "${PATH_TO_MISSING_CUSTOM_NAME_FILE}"
    SETTINGS_XDG "${XDG_DIR_WITH_MISSING_FILE}"
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
  )
  # If $XDG_DATA_HOME is set but no opencl/settings.d/cl_layer_settings.txt exists in it fall back
  # to the current working directory
  test_settings_location(
    NAME SettingsLocation-XdgMissing-Home
    EXPECTED     "cl_layer_settings.txt"
    SETTINGS_XDG "${XDG_DIR_WITH_MISSING_FILE}"
    SETTINGS_HOME
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
  )
  test_settings_location(
    NAME SettingsLocation-XdgMissing
    EXPECTED     "cl_layer_settings.txt"
    SETTINGS_XDG "${XDG_DIR_WITH_MISSING_FILE}"
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
  )
  # If $XDG_DATA_HOME is not set but HOME is, HOME should be used
  test_settings_location(
    NAME SettingsLocation-Home
    EXPECTED     "${HOME_PATH_TO_EXISTING_FILE}"
    SETTINGS_HOME
    CWD          "${DIR_WITH_EXISTING_DEFAULT_NAME_FILE1}"
  )
endif()