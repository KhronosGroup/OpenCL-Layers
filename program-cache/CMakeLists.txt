add_subdirectory(lib)

add_library (ProgramCacheLayer SHARED
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

install (
    TARGETS ProgramCacheLayer
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
