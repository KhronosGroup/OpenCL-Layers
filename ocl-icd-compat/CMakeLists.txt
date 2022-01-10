add_library (OclIcdCompatLayer SHARED
    ocl_icd_compat.c
    $<$<CXX_COMPILER_ID:MSVC>:ocl_icd_compat.def>
    $<$<CXX_COMPILER_ID:GNU>:ocl_icd_compat.map>
)

target_link_libraries (OclIcdCompatLayer PRIVATE LayersCommon)

if (NOT WIN32 AND NOT APPLE)
    set_target_properties (OclIcdCompatLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/ocl_icd_compat.map")
endif ()

install (
    TARGETS OclIcdCompatLayer
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
