add_library(ProgramCache
    src/program_cache.cpp
    src/preprocessor.cpp
    $<$<CXX_COMPILER_ID:MSVC>:src/instantiate_lexer.cpp>)

target_include_directories(ProgramCache PUBLIC inc)
target_link_libraries(ProgramCache PUBLIC LayersCommon)
target_link_libraries(ProgramCache PRIVATE Boost::wave)
if(TARGET Boost::format)
  # Boost::format is needed with newer Boost versions, because it is missing from the
  # dependency list of Boost::wave, even though it depends on it.
  target_link_libraries(ProgramCache PRIVATE Boost::format)
endif()
target_compile_features(ProgramCache PUBLIC cxx_std_17)
target_compile_definitions(ProgramCache PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:BOOST_WAVE_SEPARATE_LEXER_INSTANTIATION=0>
    $<$<CXX_COMPILER_ID:MSVC>:NOMINMAX>)
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
    TARGETS ProgramCache
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
