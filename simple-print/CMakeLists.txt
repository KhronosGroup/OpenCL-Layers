add_library (PrintLayer SHARED
    icd_print_layer.c
    icd_print_layer.h
    icd_print_layer_generated.c
    $<$<CXX_COMPILER_ID:MSVC>:icd_print_layer.def>
    $<$<CXX_COMPILER_ID:GNU>:icd_print_layer.map>
)

target_link_libraries (PrintLayer PRIVATE LayersCommon)

if (NOT WIN32 AND NOT APPLE)
    set_target_properties (PrintLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/icd_print_layer.map")
endif ()

install (TARGETS PrintLayer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
