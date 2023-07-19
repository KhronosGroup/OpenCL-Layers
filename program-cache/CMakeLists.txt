add_subdirectory(lib)

add_library (ProgramCacheLayer SHARED
    program_cache_layer_surface.cpp
    program_cache_layer.cpp
#   PLATFORM_ID taking a comma-separated list is CMake 3.15
#   $<$<AND:$<PLATFORM_ID:Windows>,$<CXX_COMPILER_ID:MSVC,Clang>>:icd_print_layer.def>
    $<$<AND:$<PLATFORM_ID:Windows>,$<OR:$<CXX_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:Clang>>>:program_cache_layer.def>
    $<$<CXX_COMPILER_ID:GNU>:program_cache_layer.map>
)

target_link_libraries(ProgramCacheLayer PRIVATE ProgramCache)

if (LAYERS_BUILD_TESTS)
    add_subdirectory(test)
endif()

set_target_properties (ProgramCacheLayer
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        PDB_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        FOLDER "Layers"
)
install (
    TARGETS ProgramCacheLayer
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
