include(CMakeParseArguments)

add_executable(print_setting_filename print_setting_filename.cpp)
target_link_libraries(print_setting_filename PRIVATE LayersUtils LayersCommon)

add_executable(print_setting_bool print_setting_bool.cpp)
target_link_libraries(print_setting_bool PRIVATE LayersUtils LayersCommon)

add_executable(print_setting_enum print_setting_enum.cpp)
target_link_libraries(print_setting_enum PRIVATE LayersUtils LayersCommon)

function(test_settings)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "NAME;SETTINGS;SETTING;SETTING_TYPE;DEFAULT;EXPECTED" "ENVIRONMENT;VARIANTS")

  string(REPLACE "." ";" SETTING_ITEMS "${ARG_SETTING}")
  list(GET SETTING_ITEMS 0 LAYER_NAME)
  list(GET SETTING_ITEMS 1 SETTING_NAME)

  set(SETTINGS_PATH "${CMAKE_CURRENT_BINARY_DIR}/test-${ARG_NAME}.txt")
  file(WRITE "${SETTINGS_PATH}" "${ARG_SETTINGS}")

  set(ENVIRONMENT "OPENCL_LAYERS_SETTINGS_PATH=${SETTINGS_PATH}")
  list(APPEND ENVIRONMENT "${ARG_ENVIRONMENT}")

  if(ARG_SETTING_TYPE STREQUAL "bool")
    set(TEST_EXE $<TARGET_FILE:print_setting_bool>)
  elseif(ARG_SETTING_TYPE STREQUAL "filename")
    set(TEST_EXE $<TARGET_FILE:print_setting_filename>)
  elseif(ARG_SETTING_TYPE STREQUAL "enum")
    set(TEST_EXE $<TARGET_FILE:print_setting_enum>)
  else()
    message(FATAL_ERROR "invalid setting type ${SETTING_TYPE}")
  endif()

  add_test(
    NAME "${ARG_NAME}"
    COMMAND
      "${TEST_EXE}" "${LAYER_NAME}" "${SETTING_NAME}" "${ARG_DEFAULT}" "${ARG_EXPECTED}" ${ARG_VARIANTS}
  )
  set_tests_properties(
    "${ARG_NAME}"
    PROPERTIES ENVIRONMENT "${ENVIRONMENT}"
  )
endfunction()

# Test retrieving the default if no config option and no override is set.
test_settings(
  NAME Settings-Default-Filename
  SETTING_TYPE filename
  SETTINGS ""
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED default
)

test_settings(
  NAME Settings-Default-Bool
  SETTING_TYPE bool
  SETTINGS ""
  SETTING test_layer.test_setting
  DEFAULT false
  EXPECTED false
)

test_settings(
  NAME Settings-Default-Enum
  SETTING_TYPE enum
  SETTINGS ""
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED default
  VARIANTS default config override
)

# Test overriding the default if no config option is set.
test_settings(
  NAME Settings-Override-Default-Filename
  SETTING_TYPE filename
  SETTINGS ""
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED override
  ENVIRONMENT OPENCL_TEST_LAYER_TEST_SETTING=override
)

test_settings(
  NAME Settings-Override-Default-Bool
  SETTING_TYPE bool
  SETTINGS ""
  SETTING test_layer.test_setting
  DEFAULT false
  EXPECTED true
  ENVIRONMENT OPENCL_TEST_LAYER_TEST_SETTING=true
)

test_settings(
  NAME Settings-Override-Default-Enum
  SETTING_TYPE enum
  SETTINGS ""
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED override
  VARIANTS default config override
  ENVIRONMENT OPENCL_TEST_LAYER_TEST_SETTING=override
)

# Test setting the value from the config.
test_settings(
  NAME Settings-Config-Filename
  SETTING_TYPE filename
  SETTINGS "test_layer.test_setting=test_setting_value"
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED test_setting_value
)

test_settings(
  NAME Settings-Config-Bool
  SETTING_TYPE bool
  SETTINGS "test_layer.test_setting=true"
  SETTING test_layer.test_setting
  DEFAULT false
  EXPECTED true
)

test_settings(
  NAME Settings-Config-Enum
  SETTING_TYPE enum
  SETTINGS "test_layer.test_setting=config"
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED config
  VARIANTS default config override
)

# Test overriding the value set from the config.
test_settings(
  NAME Settings-Override-Config
  SETTING_TYPE enum
  SETTINGS test_layer.test_setting=config
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED override
  VARIANTS default config override
  ENVIRONMENT OPENCL_TEST_LAYER_TEST_SETTING=override
)

# Test general settings file parsing.
test_settings(
  NAME Settings-Config-Comment
  SETTING_TYPE filename
  SETTINGS "#test_layer.test_setting=test_setting_value"
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED default
)

test_settings(
  NAME Settings-Config-Whitespace
  SETTING_TYPE filename
  SETTINGS "\n\ntest_layer.test_setting=test_with_whitespace"
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED test_with_whitespace
)

test_settings(
  NAME Settings-Config-Multi
  SETTING_TYPE filename
  SETTINGS "test_layer.first_setting=first_value\ntest_layer.second_setting=second_value"
  SETTING test_layer.first_setting
  DEFAULT default
  EXPECTED first_value
)

test_settings(
  NAME Settings-Config-Multi-2
  SETTING_TYPE filename
  SETTINGS "test_layer.first_setting=first_value\ntest_layer.second_setting=second_value"
  SETTING test_layer.second_setting
  DEFAULT default
  EXPECTED second_value
)

test_settings(
  NAME Settings-Override-Config-Multi-2
  SETTING_TYPE filename
  SETTINGS "test_layer.first_setting=first_value\ntest_layer.second_setting=second_value"
  SETTING test_layer.second_setting
  DEFAULT default
  EXPECTED override
  ENVIRONMENT OPENCL_TEST_LAYER_SECOND_SETTING=override
)

# Test bool values.
test_settings(
  NAME Settings-Bool-Yes
  SETTING_TYPE bool
  SETTINGS "test_layer.test_setting=yes"
  SETTING test_layer.test_setting
  DEFAULT false
  EXPECTED true
)

test_settings(
  NAME Settings-Bool-1
  SETTING_TYPE bool
  SETTINGS "test_layer.test_setting=1"
  SETTING test_layer.test_setting
  DEFAULT false
  EXPECTED true
)

test_settings(
  NAME Settings-Bool-0
  SETTING_TYPE bool
  SETTINGS "test_layer.test_setting=0"
  SETTING test_layer.test_setting
  DEFAULT true
  EXPECTED false
)

# Test invalid values
test_settings(
  NAME Settings-Bool-Invalid-False
  SETTING_TYPE bool
  SETTINGS "test_layer.test_setting=invalid"
  SETTING test_layer.test_setting
  DEFAULT false
  EXPECTED false
)

test_settings(
  NAME Settings-Bool-Invalid-True
  SETTING_TYPE bool
  SETTINGS "test_layer.test_setting=invalid"
  SETTING test_layer.test_setting
  DEFAULT true
  EXPECTED true
)

test_settings(
  NAME Settings-Enum-Invalid
  SETTING_TYPE enum
  SETTINGS "test_layer.test_setting=invalid"
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED default
  VARIANTS default config override
)

# Invalid value from environment fallback.
test_settings(
  NAME Settings-Bool-Invalid-Config
  SETTING_TYPE bool
  SETTINGS "test_layer.test_setting=invalid"
  SETTING test_layer.test_setting
  DEFAULT true
  EXPECTED true
)

test_settings(
  NAME Settings-Bool-Invalid-Override
  SETTING_TYPE bool
  SETTINGS "test_layer.test_setting=n"
  SETTING test_layer.test_setting
  DEFAULT true
  EXPECTED false
  ENVIRONMENT OPENCL_TEST_LAYER_TEST_SETTING=invalid
)

test_settings(
  NAME Settings-Enum-Invalid-Override
  SETTING_TYPE enum
  SETTINGS "test_layer.test_setting=config"
  SETTING test_layer.test_setting
  DEFAULT default
  EXPECTED config
  VARIANTS default config override
  ENVIRONMENT OPENCL_TEST_LAYER_TEST_SETTING=invalid
)

