add_library (CLObjectLifetimeLayer SHARED
    object_lifetime.cpp
    $<$<CXX_COMPILER_ID:MSVC>:object_lifetime.def>
    $<$<CXX_COMPILER_ID:GNU>:object_lifetime.map>
)

target_link_libraries (CLObjectLifetimeLayer PRIVATE LayersCommon)

if (NOT WIN32 AND NOT APPLE)
    set_target_properties (CLObjectLifetimeLayer PROPERTIES LINK_FLAGS "-Wl,--version-script -Wl,${CMAKE_CURRENT_SOURCE_DIR}/object_lifetime.map")
endif ()

install (
    TARGETS CLObjectLifetimeLayer
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
