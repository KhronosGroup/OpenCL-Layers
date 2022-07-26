add_executable (
  CLParamVerificationGenerator
  parser.cpp
)
target_link_libraries (CLParamVerificationGenerator PRIVATE RapidXml::RapidXml)

add_custom_command (
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/res.cpp
  COMMAND
    CLParamVerificationGenerator ${CMAKE_CURRENT_SOURCE_DIR}/cl-avl.xml
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/cl-avl.xml
    ${CMAKE_CURRENT_SOURCE_DIR}/param_verification.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/object_is_valid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/list_violation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/struct_violation.cpp
    CLParamVerificationGenerator
)

add_library (
  CLParamVerificationLayer
  SHARED
  param_verification.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/res.cpp
  $<$<AND:$<PLATFORM_ID:Windows>,$<OR:$<CXX_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:Clang>>>:param_verification.def>
  $<$<CXX_COMPILER_ID:GNU>:param_verification.map>
)
target_include_directories (CLParamVerificationLayer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (CLParamVerificationLayer PRIVATE LayersCommon LayersUtils)
if (NOT WIN32 AND NOT APPLE)
    set_target_properties (CLParamVerificationLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/param_verification.map")
endif ()

set (INSTALL_TARGETS CLParamVerificationLayer)
set (BUILD_TARGETS ${INSTALL_TARGETS} CLParamVerificationGenerator)

if (LAYERS_BUILD_TESTS)
  add_subdirectory (test)
endif ()

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
