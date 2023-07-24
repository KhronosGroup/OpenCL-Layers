add_library(ProgramCache "")

target_sources(ProgramCache
    PRIVATE
        $<$<AND:$<PLATFORM_ID:Windows>,$<OR:$<CXX_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:Clang>>>:src/instantiate_lexer.cpp>
        src/preprocessor.cpp
        src/preprocessor.hpp
        src/program_cache.cpp
        src/utils.hpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/common.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include/program_cache.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/program_cache/common.hpp>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/program_cache/program_cache.hpp>
)

target_include_directories(ProgramCache PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(ProgramCache
    PUBLIC
        OpenCL::Headers
    PRIVATE
        $<BUILD_INTERFACE:Boost::wave>
)
# Boost::format is needed with newer Boost versions, because it is missing from the
# dependency list of Boost::wave, even though it depends on it.
# -> CMake 3.12: TARGET_NAME_IF_EXISTS
if (TARGET Boost::format)
    target_link_libraries(ProgramCache
        PRIVATE
            $<BUILD_INTERFACE:Boost::format>
    )
endif()
target_compile_features(ProgramCache PUBLIC cxx_std_17)
target_compile_definitions(ProgramCache PRIVATE
    $<$<AND:$<PLATFORM_ID:Windows>,$<OR:$<CXX_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:Clang>>>:BOOST_WAVE_SEPARATE_LEXER_INSTANTIATION=0>
    $<$<AND:$<PLATFORM_ID:Windows>,$<OR:$<CXX_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:Clang>>>:NOMINMAX>
    CL_TARGET_OPENCL_VERSION=300
)
set_target_properties(ProgramCache PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (LAYERS_BUILD_TESTS)
    add_subdirectory(test)
endif()

set_target_properties (ProgramCache
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        FOLDER "Layers"
)
install (
    TARGETS ProgramCache EXPORT ProgramCacheTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install (
    FILES
        include/common.hpp
        include/program_cache.hpp
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/program_cache"
)
set(config_package_location "${CMAKE_INSTALL_DATADIR}/cmake/ProgramCache")
install (
    EXPORT ProgramCacheTargets
    NAMESPACE OpenCL::
    DESTINATION "${config_package_location}"
)
file(
  WRITE ${CMAKE_CURRENT_BINARY_DIR}/ProgramCache/ProgramCacheConfig.cmake
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/ProgramCacheTargets.cmake\")\n"
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ProgramCache/ProgramCacheConfig.cmake"
  DESTINATION "${config_package_location}"
)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/ProgramCache/ProgramCacheConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/ProgramCache/ProgramCacheConfigVersion.cmake
  DESTINATION ${config_package_location}
)
